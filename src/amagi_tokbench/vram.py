"""Peak VRAM by polling nvidia-smi for the whole run.

One long-lived `nvidia-smi --loop-ms` process is read on a thread. The command prefix can be
anything that ends up running nvidia-smi, so a server on another box works too:
`--nvidia-smi "ssh gpu-box nvidia-smi"`.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional

QUERY = ["--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"]


def parse_line(line: str) -> Optional[dict]:
    """'0, NVIDIA GeForce RTX 3090, 5821, 24576' -> dict, or None for anything else."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 4:
        return None
    try:
        index = int(parts[0])
        used = float(parts[-2])
        total = float(parts[-1])
    except ValueError:
        return None
    name = ", ".join(parts[1:-2]).strip()
    return {"index": index, "name": name, "used_mib": used, "total_mib": total}


def split_command(cmd: str) -> List[str]:
    """Split a command line. On Windows, backslashes in paths stay put (quote paths with spaces)."""
    if os.name == "nt":
        parts = shlex.split(cmd, posix=False)
        return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]
    return shlex.split(cmd, posix=True)


class VramSampler:
    """Track per-GPU peak memory.used while the benchmark runs."""

    def __init__(self, command: Optional[str] = "nvidia-smi", interval_ms: int = 200):
        self.command = command
        self.interval_ms = interval_ms
        self.gpus: Dict[int, dict] = {}
        self.samples = 0
        self.error: Optional[str] = None
        self._proc = None
        self._thread = None

    def available(self) -> bool:
        if not self.command:
            return False
        exe = split_command(self.command)[0]
        return shutil.which(exe) is not None

    def feed(self, line: str) -> None:
        row = parse_line(line)
        if row is None:
            return
        self.samples += 1
        gpu = self.gpus.setdefault(
            row["index"],
            {"index": row["index"], "name": row["name"], "total_mib": row["total_mib"],
             "start_mib": row["used_mib"], "peak_mib": row["used_mib"]},
        )
        gpu["peak_mib"] = max(gpu["peak_mib"], row["used_mib"])

    def start(self) -> bool:
        if not self.available():
            self.error = f"{split_command(self.command)[0] if self.command else 'nvidia-smi'} not found"
            return False
        argv = split_command(self.command) + QUERY + [f"--loop-ms={self.interval_ms}"]
        try:
            self._proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except OSError as exc:
            self.error = str(exc)
            return False
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 15
        while not self.samples and time.monotonic() < deadline and self._proc.poll() is None:
            time.sleep(0.05)
        if not self.samples:
            err = ""
            if self._proc.poll() is not None and self._proc.stderr:
                tail = self._proc.stderr.read().strip().splitlines()
                err = tail[-1] if tail else ""
            self.stop()
            self.error = err or "nvidia-smi gave no readings"
            return False
        return True

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            self.feed(line)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def result(self) -> Optional[dict]:
        if not self.gpus:
            return None
        gpus = [self.gpus[i] for i in sorted(self.gpus)]
        return {
            "gpus": gpus,
            "peak_mib": sum(g["peak_mib"] for g in gpus),
            "total_mib": sum(g["total_mib"] for g in gpus),
            "samples": self.samples,
            "source": self.command,
        }
