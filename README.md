# Free-Model Aggregator

One server that makes OpenRouter's free models a single endpoint on both
client wires. Point any OpenAI or Anthropic client at it; every answer is a
live completion from a free model, chosen per request.

## What it does

- Serves `POST /v1/chat/completions` (OpenAI wire) and `POST /v1/messages`
  (Anthropic wire), in JSON and in server-sent events, plus `GET /v1/models`
  in the shape the caller's auth header names.
- Reads the free-model catalog from the trial brief (`MAKEFUN_PROMPT_FILE`):
  each model's gateway id, the house's quality rating, and whether it takes
  tools, images and JSON mode.
- For every request it derives what the request needs (JSON mode, tools,
  images), ranks the free models that meet it by rating, and dials the best
  one on the gateway's Chat Completions route. Anthropic-shaped requests are
  translated into that body and the answer is translated back.
- A provider refusal (429 / 5xx) is retried once on the same model for a 429,
  then failed over to the next-best model that meets the request, at most
  four calls per prompt. Recovered refusals are reported on the response in
  the `x-makefun-upstream-failures` header. An answer that does not carry
  what the instruction asked for is treated as a miss and the next model is
  tried; the best answer seen is served if no model does better.
- The `model` field of every answer and every streamed chunk names the model
  that really answered, with the gateway's `openrouter/` prefix put back.

## Run

```
python3 main.py            # serve (registers with $MAKEFUN_ENV_URL in a trial)
python3 main.py --list     # print the catalog document and exit
```

Standard library only; Python 3.13. Inside a trial the server registers its
port and catalog with the env, serves the prompts, polls the env's `/work`,
and exits 0 when the env says done. Outside a trial it serves standalone,
dialing `OPENAI_BASE_URL` (default OpenRouter's public endpoint) with
`OPENAI_API_KEY` / `OPENROUTER_API_KEY`.

## License

MIT — see `LICENSE`.
