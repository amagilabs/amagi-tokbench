### Qwen2.5-32B-Instruct-Q4_K_M

llama.cpp server b11753-d8d92973c · GeForce RTX 3090

RTX 3090 24 GB, all layers on GPU (-ngl 99)

**37.0 tok/s** generation · **1,064 tok/s** prompt processing · **127 ms** time to first token · **20.6 GiB** peak vram

| case | prompt tokens | generated | TTFT | prompt tok/s | gen tok/s | server prompt tok/s | server gen tok/s |
|---|--:|--:|--:|--:|--:|--:|--:|
| short | 69 | 256 | 127 ms | 541.9 | 37.0 | 838.1 | 37.2 |
| medium | 636 | 128 | 616 ms | 1,034 | 36.8 | 1,228 | 37.0 |
| long | 2,471 | 128 | 2,323 ms | 1,064 | 36.1 | 1,208 | 36.4 |

*Median of 3 runs per case after 1 warmup. Prompt set v1, temperature 0, timed client-side on the stream. amagi-tokbench 0.1.0, 2026-10-24.* [github.com/amagilabs/amagi-tokbench](https://github.com/amagilabs/amagi-tokbench)
