"""Run the fixed prompt set against one backend and summarize it."""

from __future__ import annotations

import datetime as _dt
import secrets
import statistics
from typing import Callable, List, Optional

from . import __version__
from .backends import SEED, Backend, Sample
from .prompts import CASES, PROMPT_SET_VERSION, Case, prompt_set_hash, with_nonce

MIN_GEN_TOKENS = 32
# long/medium prompt-token ratio below this share of their word ratio means the long prompt got cut
TRUNCATION_RATIO = 0.85


def _median(values) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def summarize_case(case: Case, samples: List[Sample], max_tokens: int) -> dict:
    return {
        "name": case.name,
        "description": case.description,
        "max_tokens": max_tokens,
        "runs": [s.to_dict() for s in samples],
        "median": {
            "prompt_tokens": _median(s.prompt_tokens for s in samples),
            "gen_tokens": _median(s.gen_tokens for s in samples),
            "ttft_ms": _median(s.ttft_s * 1000 for s in samples),
            "pp_tps": _median(s.pp_tps for s in samples),
            "gen_tps": _median(s.gen_tps for s in samples),
            "server_pp_tps": _median(s.server_pp_tps for s in samples),
            "server_gen_tps": _median(s.server_gen_tps for s in samples),
        },
    }


def headline(cases: List[dict]) -> dict:
    """The four numbers on the card.

    Generation speed and time to first token come from the short case (least prompt in the way),
    prompt speed from the longest prompt that ran (least fixed overhead per token).
    """
    by_name = {c["name"]: c for c in cases}
    gen_case = by_name.get("short") or (cases[0] if cases else None)
    pp_case = max(cases, key=lambda c: c["median"]["prompt_tokens"] or 0) if cases else None
    out = {"gen_tps": None, "pp_tps": None, "ttft_ms": None, "gen_case": None, "pp_case": None}
    if gen_case:
        out.update(gen_tps=gen_case["median"]["gen_tps"], ttft_ms=gen_case["median"]["ttft_ms"],
                   gen_case=gen_case["name"])
    if pp_case:
        out.update(pp_tps=pp_case["median"]["pp_tps"], pp_case=pp_case["name"])
    return out


def check_warnings(cases: List[dict]) -> List[str]:
    warnings = []
    for c in cases:
        m = c["median"]
        if (m["gen_tokens"] or 0) < MIN_GEN_TOKENS:
            warnings.append(
                f"{c['name']}: the model stopped after {m['gen_tokens']:.0f} tokens, so its generation "
                "speed rests on few tokens"
            )
        if m["prompt_tokens"] is None:
            warnings.append(f"{c['name']}: the server did not report prompt tokens, so prompt speed is unknown")
    by_name = {c["name"]: c["median"]["prompt_tokens"] for c in cases}
    words = {c.name: c.words for c in CASES}
    expected = words["long"] / words["medium"]
    if by_name.get("long") and by_name.get("medium") and by_name["long"] < TRUNCATION_RATIO * expected * by_name["medium"]:
        warnings.append(
            "long: the prompt came back shorter than expected; the server's context window may be "
            "cutting it (Ollama: try --num-ctx 4096)"
        )
    return warnings


def run_benchmark(
    backend: Backend,
    cases: List[Case],
    runs: int = 3,
    warmup: int = 1,
    max_tokens: Optional[int] = None,
    label: Optional[str] = None,
    sampler=None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    say = progress or (lambda msg: None)
    backend.resolve_model()
    started = _dt.datetime.now(_dt.timezone.utc)
    vram = None
    if sampler is not None:
        if sampler.start():
            say(f"vram: sampling with {sampler.command}")
        else:
            say(f"vram: off ({sampler.error})")
    try:
        for i in range(warmup):
            say(f"warmup {i + 1}/{warmup} (loads the model; not counted)")
            backend.generate(with_nonce(cases[0].prompt, "warmup-" + secrets.token_hex(4)), 32)
        results = []
        for case in cases:
            n_max = max_tokens or case.max_tokens
            samples = []
            for i in range(runs):
                s = backend.generate(with_nonce(case.prompt, secrets.token_hex(4)), n_max)
                samples.append(s)
                say(
                    f"{case.name:<6} run {i + 1}/{runs}  prompt {s.prompt_tokens or '?'} tok  "
                    f"ttft {s.ttft_s * 1000:,.0f} ms  gen {s.gen_tokens} tok @ "
                    f"{_fmt(s.gen_tps)} tok/s"
                )
            results.append(summarize_case(case, samples, n_max))
    finally:
        if sampler is not None:
            sampler.stop()
            vram = sampler.result()
    info = backend.describe()
    return {
        "tool": "amagi-tokbench",
        "tool_version": __version__,
        "created": started.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "label": label,
        "backend": info,
        "vram": vram,
        "settings": {
            "prompt_set": PROMPT_SET_VERSION,
            "prompt_set_sha256": prompt_set_hash(),
            "runs": runs,
            "warmup": warmup,
            "temperature": 0,
            "seed": SEED,
            "timing": "client-side, streamed",
        },
        "cases": results,
        "summary": headline(results),
        "warnings": check_warnings(results),
    }


def _fmt(v) -> str:
    return "?" if v is None else f"{v:,.1f}"
