### Qwen/Qwen2.5-7B-Instruct-AWQ

vLLM 0.32.0 · GeForce RTX 3090

RTX 3090, vLLM 0.32.0, AWQ

**150.1 tok/s** generation · **6,485 tok/s** prompt processing · **29 ms** time to first token · **20.4 GiB** peak vram

| case | prompt tokens | generated | TTFT | prompt tok/s | gen tok/s |
|---|--:|--:|--:|--:|--:|
| short | 70 | 256 | 29 ms | 2,435 | 150.1 |
| medium | 636 | 128 | 103 ms | 6,181 | 149.2 |
| long | 2,471 | 128 | 381 ms | 6,485 | 148.1 |

*Median of 3 runs per case after 1 warmup. Prompt set v1, temperature 0, timed client-side on the stream. amagi-tokbench 0.1.0, 2026-10-26.* [github.com/amagilabs/amagi-tokbench](https://github.com/amagilabs/amagi-tokbench)
