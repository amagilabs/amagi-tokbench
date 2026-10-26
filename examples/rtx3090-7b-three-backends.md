### amagi-tokbench: 3 runs side by side

| run | backend | quant | gen tok/s | prompt tok/s | TTFT | peak VRAM |
|---|---|---|--:|--:|--:|--:|
| RTX 3090 24 GB, all layers on GPU (-ngl 99) | llama.cpp server b11753-d8d92973c | Q4_K_M | 141.9 | 4,497 | 38 ms | 5.0 GiB |
| RTX 3090, Ollama 0.40.11 | Ollama 0.40.11 | Q4_K_M | 143.5 | 4,553 | 23 ms | 4.8 GiB |
| RTX 3090, vLLM 0.32.0, AWQ | vLLM 0.32.0 | AWQ | 150.1 | 6,485 | 29 ms | 20.4 GiB |

GPU: GeForce RTX 3090.
*Prompt set v1; each number is a median from its own run.* [github.com/amagilabs/amagi-tokbench](https://github.com/amagilabs/amagi-tokbench)
