---
name: amagi-tokbench
description: Benchmark a local LLM server (Ollama, llama.cpp server, vLLM or any OpenAI-compatible endpoint) with a fixed prompt set and report generation tok/s, prompt-processing tok/s, time to first token and peak VRAM as a markdown table and a PNG card. Use when asked how fast a local model runs, to compare two models, quants, backends or GPUs, to check whether a model fits in VRAM, or to make a shareable speed card for a local LLM setup.
---

# amagi-tokbench

A CLI that sends the same three prompts to a running local LLM server, times the streamed reply
on the client, and writes `PREFIX.json`, `PREFIX.md` and `PREFIX.png`. It does not start or stop
servers and does not load models. It benchmarks whatever is already running.

## When to use this skill

- The user asks how many tokens per second their local model gets.
- The user wants to compare models, quants, context sizes, backends or GPUs on equal terms.
- The user wants to know how much VRAM a model takes while it is generating.
- The user wants a card or table to paste into an issue, a README or a forum post.

## How to use it

Install once: `pipx install amagi-tokbench` (or `pip install -e .` from this repo in development).

```bash
amagi-tokbench detect                          # which server is listening, which model is loaded
amagi-tokbench run                             # auto-detect on localhost, all three cases
amagi-tokbench run --url http://localhost:8080 --label "RTX 3090, -ngl 99" -o 3090-qwen7b
amagi-tokbench run --backend ollama -m qwen2.5:7b --num-ctx 4096
amagi-tokbench run --nvidia-smi "ssh gpu-box nvidia-smi"   # server runs on another machine
amagi-tokbench card 3090-qwen7b.json           # re-render the card from saved results
amagi-tokbench compare a.json b.json -o a-vs-b # one table and PNG for several runs
amagi-tokbench prompts --show long             # print the fixed prompts
```

Auto-detect tries ports 11434 (Ollama), 8080 (llama.cpp), 8000 (vLLM) and 1234 (OpenAI-compatible)
on localhost. Pass `--url` and `--backend` when the server is elsewhere or detection guesses wrong.
`--api-key` (or `TOKBENCH_API_KEY`) sends a bearer token. Never write a key into a file or a
command you show back to the user; prefer the environment variable.

Exit codes: 0 ok, 2 usage error (bad option, not a results file), 3 server error (nothing
listening, model not loaded, request failed).

## What the numbers mean

- **gen tok/s**: tokens after the first one, divided by the time from first to last streamed
  token. The headline uses the `short` case.
- **prompt tok/s**: prompt tokens divided by time to first token, so it includes request overhead
  and the first decode step. The headline uses the longest case. On llama.cpp and Ollama the card
  also shows the server's own figures, which are higher because they exclude that overhead.
- **TTFT**: time from sending the request to the first streamed text.
- **peak VRAM**: highest `memory.used` that `nvidia-smi` reported on the GPU during the run. It
  counts everything on that GPU, not just this model. If another process shares the GPU, say so.

Each case runs `--runs` times (default 3) after `--warmup` untimed requests (default 1), and the
card shows the median. Every prompt starts with a random run tag so a prefix cache cannot answer
it. Temperature is 0 and generation is forced to the token limit where the server allows it.

## Working with the results

1. Read `summary` (`gen_tps`, `pp_tps`, `ttft_ms`) and `vram.peak_mib` / `vram.total_mib` from the
   JSON. Per-case medians are under `cases[].median`.
2. Read `warnings` before you quote any number. "the prompt came back shorter than expected" means
   the server's context window probably cut the long prompt (on Ollama, re-run with
   `--num-ctx 4096`), so its prompt rate is not valid until that is fixed. "the model stopped after
   N tokens" means the generation rate rests on too few tokens.
3. When comparing, only compare runs with the same `settings.prompt_set`. `compare` adds a note
   when the prompt sets differ.
4. Quote numbers exactly as the card shows them. Do not round them up, average across machines, or
   present one machine's result as typical for a GPU.

## Privacy

Results files keep the model's file name but drop local directory paths. They do not store the
server URL or the `--nvidia-smi` command. The `--label` text goes into the card as written, so keep host names and personal
details out of it if the card will be shared.

## Limits

- VRAM needs an NVIDIA GPU and `nvidia-smi`. Without it the speed numbers still work and the card
  shows no VRAM figure.
- Token counts come from the server (`usage`, llama.cpp `timings`, or Ollama counters). An
  OpenAI-compatible server that sends none falls back to counting stream chunks, and the card
  shows no prompt rate.
- One request at a time. It measures single-user latency and speed, not batched throughput.
