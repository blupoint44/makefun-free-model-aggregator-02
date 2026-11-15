#!/usr/bin/env python3
"""One server, two wires, live free models behind every answer.

An aggregator: a client points an OpenAI-shaped or Anthropic-shaped
request at this server, the server picks the best free model for the
request's shape from the brief's catalog, dials it on the gateway's
Chat Completions route, and translates the answer back into the wire
the client used. Refusals (429 / 5xx) are retried or failed over to
the next-best model that meets the request, and reported on the
response in the x-makefun-upstream-failures header.

Run inside a trial it registers its port and catalog with the env,
serves until the env's /work says done, then exits 0.
Run with --list it prints the catalog document and exits 0.
Standard library only.
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------- brief

MODELS_SCHEMA = "free-model-aggregator-02/models/v1"
PROVIDER_ID = "openrouter"
PROVIDER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER_KEY_ENV = "OPENROUTER_API_KEY"
PROVIDER_WIRE = "openai"

DEFAULT_MAX_TOKENS = 1024
MAX_ATTEMPTS_PER_PROMPT = 4
RETRY_429_DELAY_SEC = 2.0
UPSTREAM_TIMEOUT_SEC = 12.0
TOKEN_RE = re.compile(r"[A-Z]{3,12}-[0-9]{4}")


class Model:
    """One row of the brief's free-model catalog."""

    def __init__(self, row, index):
        self.index = index
        self.id = row["id"]                      # dialable id, openrouter/...
        self.display_name = row.get("display_name") or row["id"]
        self.rating = int(row.get("rating_milli") or 0)
        self.context_length = row.get("context_length")
        self.tools = bool(row.get("tools"))
        self.vision = bool(row.get("vision"))
        self.json = bool(row.get("json"))
        self.free = bool(row.get("free", True))

    def meets(self, needs):
        if needs.get("json") and not self.json:
            return False
        if needs.get("tools") and not self.tools:
            return False
        if needs.get("vision") and not self.vision:
            return False
        return True


def _json_objects(text):
    """Yield every top-level JSON object embedded in a text document."""
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            return
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(value, dict):
            yield value
        pos = end


def load_brief(path):
    """Read the trial brief: the free-model catalog it carries."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    models = []
    for obj in _json_objects(text):
        if obj.get("schema") == MODELS_SCHEMA and isinstance(obj.get("models"), list):
            for index, row in enumerate(obj["models"]):
                if isinstance(row, dict) and isinstance(row.get("id"), str):
                    models.append(Model(row, index))
            break
    return models


def build_catalog(models):
    """The catalog document: the same object --list prints and we register."""
    rows = []
    for model in models:
        if not model.free:
            continue
        rows.append({
            "id": model.id,
            "free": True,
            "context_length": model.context_length,
        })
    return {
        "providers": [{
            "id": PROVIDER_ID,
            "base_url": PROVIDER_BASE_URL,
            "key_env": PROVIDER_KEY_ENV,
            "wire": PROVIDER_WIRE,
            "models": rows,
        }]
    }


def catalog_text(models):
    return json.dumps(build_catalog(models), separators=(",", ":"))


# ------------------------------------------------------- request shaping

def anthropic_system_text(system):
    """An Anthropic system field (string or text blocks) as one string."""
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return None


def anthropic_content_to_openai(content):
    """Translate one Anthropic message's content into OpenAI parts.

    Returns (text_or_parts, tool_calls, tool_messages): the message's own
    content as a string, a list of parts, or None; any tool calls the
    assistant made; and any tool-result messages a user turn carries.
    """
    if content is None or isinstance(content, str):
        return content, [], []
    if not isinstance(content, list):
        return str(content), [], []
    parts = []
    tool_calls = []
    tool_messages = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64":
                url = "data:%s;base64,%s" % (source.get("media_type", "image/png"),
                                             source.get("data", ""))
            else:
                url = str(source.get("url", ""))
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif kind == "tool_use":
            tool_calls.append({
                "id": str(block.get("id", "")),
                "type": "function",
                "function": {
                    "name": str(block.get("name", "")),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })
        elif kind == "tool_result":
            inner = block.get("content")
            if isinstance(inner, list):
                text = "".join(
                    str(b.get("text", "")) for b in inner
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            elif inner is None:
                text = ""
            else:
                text = str(inner)
            tool_messages.append({
                "role": "tool",
                "tool_call_id": str(block.get("tool_use_id", "")),
                "content": text,
            })
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"], tool_calls, tool_messages
    if not parts:
        return None, tool_calls, tool_messages
    return parts, tool_calls, tool_messages


def anthropic_to_openai(body):
    """Translate an Anthropic Messages body into a Chat Completions body."""
    out = {"messages": []}
    system = anthropic_system_text(body.get("system"))
    if system:
        out["messages"].append({"role": "system", "content": system})
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content, tool_calls, tool_messages = anthropic_content_to_openai(
            message.get("content"))
        for tool_message in tool_messages:
            out["messages"].append(tool_message)
        if role == "assistant":
            entry = {"role": "assistant", "content": content}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out["messages"].append(entry)
        elif content is not None or not tool_messages:
            out["messages"].append({"role": "user", "content": content})
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("stop_sequences") is not None:
        out["stop"] = body["stop_sequences"]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = [{
            "type": "function",
            "function": {
                "name": str(tool.get("name", "")),
                "description": str(tool.get("description", "")),
                "parameters": tool.get("input_schema") or {"type": "object"},
            },
        } for tool in tools if isinstance(tool, dict)]
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "any":
            out["tool_choice"] = "required"
        elif kind == "none":
            out["tool_choice"] = "none"
        elif kind == "tool" and choice.get("name"):
            out["tool_choice"] = {
                "type": "function",
                "function": {"name": str(choice["name"])},
            }
        else:
            out["tool_choice"] = "auto"
    if body.get("stream"):
        out["stream"] = True
    return out


# ------------------------------------------------------------- selection

def request_needs(chat_body):
    """What a Chat Completions body demands of the model behind it."""
    needs = {}
    if chat_body.get("response_format") is not None:
        needs["json"] = True
    if isinstance(chat_body.get("tools"), list) and chat_body["tools"]:
        needs["tools"] = True
    for message in chat_body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                    needs["vision"] = True
    return needs


def ranked_candidates(models, needs):
    """Free models meeting the request, best first (brief order breaks ties)."""
    pool = [m for m in models if m.free and m.meets(needs)]
    pool.sort(key=lambda m: (-m.rating, m.index))
    return pool


def expected_token(chat_body):
    """The one token the instruction asks back, if it names exactly one."""
    found = []
    for message in chat_body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
        for text in texts:
            found.extend(TOKEN_RE.findall(text))
    distinct = set(found)
    if len(distinct) == 1:
        return found[0]
    return None


def estimate_tokens(text):
    return max(1, len(text or "") // 4)


def request_text(chat_body):
    parts = []
    for message in chat_body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
    return "\n".join(parts)


# -------------------------------------------------------------- gateway

class Refusal(Exception):
    """The gateway or the provider refused a call."""

    def __init__(self, status, code, body_text):
        super().__init__("upstream refusal %s" % status)
        self.status = status
        self.code = code
        self.body_text = body_text


def refusal_code(body_text):
    try:
        value = json.loads(body_text)
    except (ValueError, TypeError):
        return ""
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        error = value["error"]
        code = error.get("code")
        if isinstance(code, str) and code:
            return code[:64]
        kind = error.get("type")
        if isinstance(kind, str):
            return kind[:64]
    return ""


class Gateway:
    """Dials the trial gateway's Chat Completions route on the trial key."""

    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def call(self, chat_body):
        """One call. Returns (status, body bytes) for any HTTP answer,
        or raises Refusal for transport-level failures."""
        url = self.base_url + "/chat/completions"
        data = json.dumps(chat_body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Authorization", "Bearer " + self.api_key)
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_SEC) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()
        except (urllib.error.URLError, OSError) as err:
            raise Refusal(503, "upstream_unreachable", str(err))


# ------------------------------------------------------- answer assembly

def openai_message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def parse_openai_json(raw):
    """A provider's non-streaming Chat Completions answer, or None."""
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        return None
    return value


def parse_openai_stream(raw):
    """Assemble a provider's streamed answer: text, tool calls, the
    finish reason, the usage and the chunk skeleton to answer with."""
    text = raw.decode("utf-8", "replace")
    chunks = []
    content_parts = []
    tool_calls = {}
    finish_reason = None
    usage = None
    chunk_id = None
    created = 0
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(chunk, dict):
            continue
        chunks.append(chunk)
        if chunk_id is None and isinstance(chunk.get("id"), str):
            chunk_id = chunk["id"]
        if isinstance(chunk.get("created"), int):
            created = chunk["created"]
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        if isinstance(choice.get("finish_reason"), str):
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
        if isinstance(delta.get("tool_calls"), list):
            for call in delta["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                index = call.get("index", 0)
                slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if isinstance(call.get("id"), str):
                    slot["id"] = call["id"]
                function = call.get("function")
                if isinstance(function, dict):
                    if isinstance(function.get("name"), str):
                        slot["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        slot["arguments"] += function["arguments"]
    ordered_tools = [tool_calls[k] for k in sorted(tool_calls)]
    return {
        "chunks": chunks,
        "text": "".join(content_parts),
        "tool_calls": ordered_tools,
        "finish_reason": finish_reason,
        "usage": usage,
        "id": chunk_id,
        "created": created,
    }


def whole_usage(usage, chat_body, answer_text):
    """The provider's usage when it is whole, else a deterministic estimate."""
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int) and completion >= 1:
            return {"prompt_tokens": prompt, "completion_tokens": completion,
                    "total_tokens": prompt + completion}
    prompt = estimate_tokens(request_text(chat_body))
    completion = estimate_tokens(answer_text)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def deterministic_id(prefix, seed_text):
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:24]
    return prefix + digest


def openai_to_anthropic(answer, dialed_id):
    """Translate a provider's Chat Completions answer to a Messages one."""
    choice = answer["choices"][0]
    message = choice["message"]
    content = []
    text = openai_message_text(message)
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        raw_args = function.get("arguments")
        try:
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else {}
            if not isinstance(parsed_args, dict):
                parsed_args = {"value": parsed_args}
        except ValueError:
            parsed_args = {}
        content.append({
            "type": "tool_use",
            "id": str(call.get("id") or deterministic_id("toolu_", str(function.get("name", "")))),
            "name": str(function.get("name", "")),
            "input": parsed_args,
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = answer.get("usage") if isinstance(answer.get("usage"), dict) else {}
    finish = choice.get("finish_reason") or "stop"
    return {
        "id": deterministic_id("msg_", str(answer.get("id") or text)),
        "type": "message",
        "role": "assistant",
        "model": dialed_id,
        "content": content,
        "stop_reason": FINISH_TO_STOP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
        },
    }


# --------------------------------------------------------------- serving

class Served:
    """The outcome of serving one prompt: what to send the client."""

    def __init__(self, status, content_type, body, failures):
        self.status = status
        self.content_type = content_type
        self.body = body                    # bytes
        self.failures = failures            # refusal trail rows


def emit_openai_stream(assembled, dialed_id, chat_body):
    """A normalized OpenAI event stream built from the provider's chunks:
    its content chunks with the dialed id put back, one finish chunk, one
    usage chunk, then [DONE]."""
    lines = []
    for chunk in assembled["chunks"]:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue                    # usage-only chunk: re-emitted whole below
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        if isinstance(choice.get("finish_reason"), str):
            continue                    # the finish chunk is re-emitted once below
        out = dict(chunk)
        out["model"] = dialed_id
        lines.append("data: " + json.dumps(out, separators=(",", ":")))
    finish_reason = assembled["finish_reason"] or "stop"
    base = {
        "id": assembled["id"] or deterministic_id("chatcmpl-", assembled["text"]),
        "object": "chat.completion.chunk",
        "created": assembled["created"],
        "model": dialed_id,
    }
    finish_chunk = dict(base)
    finish_chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
    lines.append("data: " + json.dumps(finish_chunk, separators=(",", ":")))
    usage_chunk = dict(base)
    usage_chunk["choices"] = []
    usage_chunk["usage"] = whole_usage(assembled["usage"], chat_body, assembled["text"])
    lines.append("data: " + json.dumps(usage_chunk, separators=(",", ":")))
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


def emit_anthropic_stream(assembled, dialed_id, chat_body):
    """The canonical Anthropic event sequence for an assembled answer."""
    usage = whole_usage(assembled["usage"], chat_body, assembled["text"])
    message_id = deterministic_id("msg_", assembled["id"] or assembled["text"])
    events = []

    def event(name, payload):
        events.append("event: %s\ndata: %s" % (name, json.dumps(payload, separators=(",", ":"))))

    event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": dialed_id,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": 0},
        },
    })
    index = 0
    text = assembled["text"]
    if text:
        event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "text", "text": ""},
        })
        step = max(1, len(text) // 6)
        for at in range(0, len(text), step):
            event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": text[at:at + step]},
            })
        event("content_block_stop", {"type": "content_block_stop", "index": index})
        index += 1
    for call in assembled["tool_calls"]:
        event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {
                "type": "tool_use",
                "id": call["id"] or deterministic_id("toolu_", call["name"]),
                "name": call["name"],
                "input": {},
            },
        })
        if call["arguments"]:
            step = max(1, len(call["arguments"]) // 4)
            for at in range(0, len(call["arguments"]), step):
                event("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": call["arguments"][at:at + step]},
                })
        event("content_block_stop", {"type": "content_block_stop", "index": index})
        index += 1
    if index == 0:
        event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        event("content_block_stop", {"type": "content_block_stop", "index": 0})
    finish_reason = assembled["finish_reason"] or ("tool_calls" if assembled["tool_calls"] else "stop")
    event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": FINISH_TO_STOP.get(finish_reason, "end_turn"),
                  "stop_sequence": None},
        "usage": {"output_tokens": usage["completion_tokens"]},
    })
    event("message_stop", {"type": "message_stop"})
    return ("\n\n".join(events) + "\n\n").encode("utf-8")


def serve_completion(wire, client_body, models, gateway):
    """Serve one prompt on one wire. wire is 'openai' or 'anthropic'."""
    if wire == "anthropic":
        chat_body = anthropic_to_openai(client_body)
    else:
        chat_body = dict(client_body)
        chat_body.pop("model", None)
    stream = bool(client_body.get("stream"))
    if chat_body.get("max_tokens") is None and chat_body.get("max_completion_tokens") is None:
        chat_body["max_tokens"] = DEFAULT_MAX_TOKENS
    needs = request_needs(chat_body)
    token = expected_token(chat_body)
    candidates = ranked_candidates(models, needs)

    failures = []
    fallback = None          # (rank, model, kind, payload) best answer without the token
    attempts = 0
    for rank, model in enumerate(candidates):
        if attempts >= MAX_ATTEMPTS_PER_PROMPT:
            break
        retried_429 = False
        while attempts < MAX_ATTEMPTS_PER_PROMPT:
            attempts += 1
            dial = dict(chat_body)
            dial["model"] = model.id
            if stream:
                dial["stream"] = True
                options = dict(dial.get("stream_options") or {})
                options["include_usage"] = True
                dial["stream_options"] = options
            try:
                status, raw = gateway.call(dial)
            except Refusal as refusal:
                status, raw = refusal.status, refusal.body_text.encode("utf-8")
            if status < 200 or status >= 300:
                code = refusal_code(raw.decode("utf-8", "replace"))
                failures.append({"status": status, "code": code, "model": model.id})
                if status == 429 and not retried_429 and attempts < MAX_ATTEMPTS_PER_PROMPT:
                    retried_429 = True
                    time.sleep(RETRY_429_DELAY_SEC)
                    continue
                break
            if stream:
                payload = parse_openai_stream(raw)
                text = payload["text"]
                usable = True
            else:
                payload = parse_openai_json(raw)
                if payload is None:
                    failures.append({"status": 502, "code": "bad_upstream_body",
                                     "model": model.id})
                    break
                text = openai_message_text(payload["choices"][0]["message"])
                usable = True
            if token is not None and token not in text:
                usable = False
            if usable:
                return render(wire, stream, payload, model.id, chat_body, failures)
            if fallback is None:
                fallback = (model, payload)
            break
    if fallback is not None:
        model, payload = fallback
        return render(wire, stream, payload, model.id, chat_body, failures)
    status = failures[-1]["status"] if failures else 502
    if status < 400 or status > 599:
        status = 502
    return render_error(wire, status, failures)


def render(wire, stream, payload, dialed_id, chat_body, failures):
    if stream:
        if wire == "anthropic":
            body = emit_anthropic_stream(payload, dialed_id, chat_body)
        else:
            body = emit_openai_stream(payload, dialed_id, chat_body)
        return Served(200, "text/event-stream", body, failures)
    if wire == "anthropic":
        answer = openai_to_anthropic(payload, dialed_id)
        answer["usage"] = {
            "input_tokens": whole_usage(payload.get("usage"), chat_body,
                                        openai_message_text(payload["choices"][0]["message"]))["prompt_tokens"],
            "output_tokens": whole_usage(payload.get("usage"), chat_body,
                                         openai_message_text(payload["choices"][0]["message"]))["completion_tokens"],
        }
        return Served(200, "application/json",
                      json.dumps(answer, separators=(",", ":")).encode("utf-8"), failures)
    answer = dict(payload)
    answer["model"] = dialed_id
    answer["usage"] = whole_usage(payload.get("usage"), chat_body,
                                  openai_message_text(payload["choices"][0]["message"]))
    return Served(200, "application/json",
                  json.dumps(answer, separators=(",", ":")).encode("utf-8"), failures)


def render_error(wire, status, failures):
    if wire == "anthropic":
        body = {"type": "error",
                "error": {"type": "api_error",
                          "message": "every free model refused this request"}}
    else:
        body = {"error": {"message": "every free model refused this request",
                          "type": "upstream_error", "code": "upstream_refused"}}
    return Served(status, "application/json",
                  json.dumps(body, separators=(",", ":")).encode("utf-8"), failures)


# ------------------------------------------------------------ HTTP front

class Context:
    def __init__(self, models, gateway):
        self.models = models
        self.gateway = gateway


def models_document(wire, models):
    rows = [m for m in models if m.free]
    if wire == "anthropic":
        data = [{
            "type": "model",
            "id": m.id,
            "display_name": m.display_name,
            "created_at": "2026-01-01T00:00:00Z",
            "free": True,
        } for m in rows]
        return {
            "data": data,
            "has_more": False,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }
    return {
        "object": "list",
        "data": [{
            "id": m.id,
            "object": "model",
            "created": 1767225600,
            "owned_by": "openrouter",
            "free": True,
        } for m in rows],
    }


def make_handler(context):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, status, content_type, body, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, status, value):
            self._send(status, "application/json",
                       json.dumps(value, separators=(",", ":")).encode("utf-8"))

        def do_GET(self):
            if self.path.split("?")[0] == "/v1/models":
                if self.headers.get("x-api-key") is not None and \
                        self.headers.get("Authorization") is None:
                    self._send_json(200, models_document("anthropic", context.models))
                else:
                    self._send_json(200, models_document("openai", context.models))
                return
            self._send_json(404, {"error": {"message": "no such route",
                                            "type": "invalid_request_error"}})

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/v1/chat/completions", "/v1/messages"):
                self._send_json(404, {"error": {"message": "no such route",
                                                "type": "invalid_request_error"}})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("body is not an object")
            except (ValueError, UnicodeDecodeError):
                self._send_json(400, {"error": {"message": "the body is not JSON",
                                                "type": "invalid_request_error"}})
                return
            wire = "openai" if path == "/v1/chat/completions" else "anthropic"
            served = serve_completion(wire, body, context.models, context.gateway)
            extra = {}
            if served.failures:
                extra["x-makefun-upstream-failures"] = json.dumps(
                    served.failures[:8], separators=(",", ":"))
            self._send(served.status, served.content_type, served.body, extra)

    return Handler


# ------------------------------------------------------------ trial glue

def get_json(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def register_with_env(env_url, port, models):
    body = ('{"protocol_version":1,"port":%d,"catalog":%s}'
            % (port, catalog_text(models))).encode("utf-8")
    deadline = time.time() + 60.0
    while time.time() < deadline:
        try:
            request = urllib.request.Request(
                env_url.rstrip("/") + "/register", data=body, method="POST")
            request.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(request, timeout=5.0) as response:
                answer = json.loads(response.read().decode("utf-8"))
            if isinstance(answer, dict) and answer.get("ok"):
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def poll_until_done(env_url, server):
    url = env_url.rstrip("/") + "/work"
    while True:
        try:
            state = get_json(url)
            if isinstance(state, dict) and state.get("done"):
                break
        except Exception:
            pass
        time.sleep(0.3)
    threading.Thread(target=server.shutdown, daemon=True).start()


def main(argv):
    models = []
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    if prompt_file and os.path.exists(prompt_file):
        try:
            models = load_brief(prompt_file)
        except Exception:
            models = []
    if "--list" in argv:
        sys.stdout.write(catalog_text(models) + "\n")
        return 0
    base_url = os.environ.get("OPENAI_BASE_URL") or PROVIDER_BASE_URL
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(PROVIDER_KEY_ENV) or ""
    context = Context(models, Gateway(base_url, api_key))
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(context))
    server.daemon_threads = True
    port = server.server_address[1]
    env_url = os.environ.get("MAKEFUN_ENV_URL")
    if env_url:
        if register_with_env(env_url, port, models):
            threading.Thread(target=poll_until_done, args=(env_url, server),
                             daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
