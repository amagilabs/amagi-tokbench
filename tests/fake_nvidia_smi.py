"""Stands in for `nvidia-smi --query-gpu=... --loop-ms=N`: prints a rising memory.used reading."""

import sys
import time

loop_ms = 100
for arg in sys.argv[1:]:
    if arg.startswith("--loop-ms="):
        loop_ms = int(arg.split("=", 1)[1])

used = 4000
while True:
    print(f"0, NVIDIA GeForce RTX 3090, {used}, 24576", flush=True)
    used = min(used + 100, 5200)
    time.sleep(loop_ms / 1000)
