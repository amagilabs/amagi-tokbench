### qwen2.5-7b-q4km:latest

Ollama 0.40.11 · Q4_K_M · 7.6B · GeForce RTX 3090

RTX 3090, Ollama 0.40.11

**143.5 tok/s** generation · **4,553 tok/s** prompt processing · **23 ms** time to first token · **4.8 GiB** peak vram

| case | prompt tokens | generated | TTFT | prompt tok/s | gen tok/s | server prompt tok/s | server gen tok/s |
|---|--:|--:|--:|--:|--:|--:|--:|
| short | 68 | 144 | 23 ms | 2,968 | 143.5 | 3,885 | 143.5 |
| medium | 637 | 128 | 137 ms | 4,667 | 141.4 | 5,158 | 142.6 |
| long | 2,470 | 128 | 542 ms | 4,553 | 137.8 | 5,195 | 138.9 |

*Median of 3 runs per case after 1 warmup. Prompt set v1, temperature 0, timed client-side on the stream. amagi-tokbench 0.1.0, 2026-10-26.* [github.com/amagilabs/amagi-tokbench](https://github.com/amagilabs/amagi-tokbench)
