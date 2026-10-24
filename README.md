# amagi-tokbench

One command that measures how fast your local LLM runs on Ollama, llama.cpp server or vLLM: generation tok/s, prompt-processing tok/s, time to first token and peak VRAM, from the same fixed prompts every time. It writes a markdown table and a PNG card you can paste into an issue or a forum thread.

![amagi-tokbench rendering cards from two saved runs](https://raw.githubusercontent.com/amagilabs/amagi-tokbench/master/assets/demo.gif)

## Install

```bash
pipx install amagi-tokbench
```

Not on PyPI yet. Until it is, install from a clone with `pipx install .` or `pip install .`

Python 3.9 or newer. The only dependency is Pillow, for the PNG.

## Usage

Start your server and load a model the usual way, then run:

```bash
amagi-tokbench run
```

It looks for a server on localhost ports 11434 (Ollama), 8080 (llama.cpp), 8000 (vLLM) and 1234 (any OpenAI-compatible server), uses the model that is loaded, runs three prompts three times each after one warmup request, and writes `tokbench.json`, `tokbench.md` and `tokbench.png`.

```bash
amagi-tokbench run --url http://localhost:8080 --label "RTX 3090, -ngl 99" -o 3090-qwen7b
amagi-tokbench run --backend ollama -m qwen2.5:7b --num-ctx 4096
amagi-tokbench detect          # which server answered, and what it has loaded
amagi-tokbench prompts         # the fixed prompt set
```

## A real run

Qwen2.5 7B Instruct, Q4_K_M, on llama.cpp server build b11753 with every layer on one RTX 3090:

![card for Qwen2.5-7B Q4_K_M on an RTX 3090](https://raw.githubusercontent.com/amagilabs/amagi-tokbench/master/examples/rtx3090-llamacpp-qwen2.5-7b-q4km.png)

| case | prompt tokens | generated | TTFT | prompt tok/s | gen tok/s | server prompt tok/s | server gen tok/s |
|---|--:|--:|--:|--:|--:|--:|--:|
| short | 70 | 256 | 38 ms | 1,865 | 141.9 | 3,213 | 142.3 |
| medium | 637 | 128 | 154 ms | 4,132 | 140.9 | 5,056 | 142.0 |
| long | 2,470 | 128 | 549 ms | 4,497 | 137.3 | 5,135 | 138.4 |

Peak VRAM was 5.0 of 24.0 GiB. The same GPU running Qwen2.5 32B at Q4_K_M did 37.0 gen tok/s and 1,064 prompt tok/s, with a 127 ms first token and 20.6 GiB of VRAM in use. The JSON, markdown and PNG for both runs are in [`examples/`](examples/).

Generation speed measured on the client lands within 1% of what llama.cpp reports for itself. Prompt speed measured on the client is lower than the server's figure because it also counts the HTTP round trip and the first decode step, and that is part of the wait a user sees. The card shows both when the server reports its own.

## Compare runs

```bash
amagi-tokbench compare 3090-qwen7b.json 3090-qwen32b.json -o 7b-vs-32b
```

`compare` puts several results files in one table and one PNG with bar charts. Columns that read the same for every run, such as the backend, leave the table and go on one shared line.

![7B and 32B on the same RTX 3090](https://raw.githubusercontent.com/amagilabs/amagi-tokbench/master/examples/rtx3090-7b-vs-32b.png)

`amagi-tokbench card results.json` re-renders the card for one saved run. The demo GIF above shows `prompts`, `card` and `compare` running on the two saved RTX 3090 results.

## How it measures

- **Three fixed prompts**, the same on every machine: `short` (23 words, up to 256 new tokens), `medium` (399 words, about 600 prompt tokens, 128 new) and `long` (1,607 words, about 2,500 prompt tokens, 128 new). `amagi-tokbench prompts --show long` prints one in full. The prompt set has a version and a hash, and both go in the results file.
- **Timing on the client.** It streams the reply and times each chunk as it arrives. TTFT is the time from sending the request to the first text. Gen tok/s counts the tokens after the first one over the time from first to last. Prompt tok/s is prompt tokens divided by TTFT.
- **The card headline** takes generation speed and TTFT from `short` and prompt speed from the longest prompt. Each number is the median of the timed runs.
- **No cache hits.** Each prompt starts with a random run tag, so a prefix cache can't answer it. On llama.cpp it also sends `cache_prompt: false`.
- **Same length every time.** Temperature 0, seed 42. On llama.cpp and vLLM it sets `ignore_eos`, so the model writes the full token budget.
- **Peak VRAM** is the highest `memory.used` that `nvidia-smi` reported during the run, sampled every 200 ms. It covers the whole GPU, so close anything else using it first.

If the server's context window cuts off the long prompt, which Ollama's default context can do, the run prints a warning that names the fix (`--num-ctx 4096`). It also warns when a model stops early or the server reports no prompt token count.

## Options

```
amagi-tokbench run [--backend auto|ollama|llamacpp|vllm|openai] [--url URL] [-m MODEL]
                   [--api-key KEY] [--runs N] [--warmup N] [--cases short,medium,long]
                   [--max-tokens N] [--label TEXT] [-o PREFIX] [--no-png]
                   [--nvidia-smi CMD] [--no-vram] [--num-ctx N] [--timeout S] [-q]
```

- `--nvidia-smi "ssh gpu-box nvidia-smi"` reads VRAM from the machine the server runs on, when that isn't the machine you run the benchmark from.
- `--api-key` sends a bearer token. You can set `TOKBENCH_API_KEY` instead so the key stays out of your shell history.
- Exit codes: 0 ok, 2 usage error, 3 the server failed or nothing answered.

## What goes in the results file

The model's file name, the backend and its version, the GPU name, the settings and the numbers. Local folder paths are cut from model names, and neither the server URL nor your `--nvidia-smi` command is stored. Your `--label` text is copied as written.

## Use it from Claude Code

This repo ships a skill at `.claude/skills/amagi-tokbench/SKILL.md`. Copy that folder into your own `.claude/skills/` and Claude Code can run the benchmark, read the results and compare runs with you.

## Limits

- The llama.cpp numbers above come from a real server. Ollama, vLLM and the generic OpenAI-compatible path are covered by tests against fake servers that speak their streaming formats. They have not yet been run against real Ollama or vLLM servers.
- VRAM needs an NVIDIA GPU and `nvidia-smi`. On other GPUs the speed numbers still work and the card shows no VRAM figure.
- One request at a time. It measures what a single user sees, not batched throughput.
- An OpenAI-compatible server that reports no token counts gets its generated tokens counted by stream chunk, and no prompt speed.

## If this saved you time

A star helps other people find it.

## More from AmagiLabs

[github.com/amagilabs](https://github.com/amagilabs) has the other small open source dev tools from the same lab.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Amagisolutionsgroup, Inc. (AmagiLabs)

The PNG cards use the Montserrat font, bundled under the SIL Open Font License; see [`src/amagi_tokbench/vendor/fonts/OFL.txt`](src/amagi_tokbench/vendor/fonts/OFL.txt).
