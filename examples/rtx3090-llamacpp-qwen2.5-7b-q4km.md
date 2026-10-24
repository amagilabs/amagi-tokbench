### Qwen2.5-7B-Instruct-Q4_K_M

llama.cpp server b11753-d8d92973c · GeForce RTX 3090

RTX 3090 24 GB, all layers on GPU (-ngl 99)

**141.9 tok/s** generation · **4,497 tok/s** prompt processing · **38 ms** time to first token · **5.0 GiB** peak vram

| case | prompt tokens | generated | TTFT | prompt tok/s | gen tok/s | server prompt tok/s | server gen tok/s |
|---|--:|--:|--:|--:|--:|--:|--:|
| short | 70 | 256 | 38 ms | 1,865 | 141.9 | 3,213 | 142.3 |
| medium | 637 | 128 | 154 ms | 4,132 | 140.9 | 5,056 | 142.0 |
| long | 2,470 | 128 | 549 ms | 4,497 | 137.3 | 5,135 | 138.4 |

*Median of 3 runs per case after 1 warmup. Prompt set v1, temperature 0, timed client-side on the stream. amagi-tokbench 0.1.0, 2026-10-24.* [github.com/amagilabs/amagi-tokbench](https://github.com/amagilabs/amagi-tokbench)
