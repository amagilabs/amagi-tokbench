"""amagi-tokbench command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .backends import BACKEND_LABELS, BACKENDS, BenchError, find_server, make_backend
from .bench import run_benchmark
from .card import compare_markdown, compare_text, render_compare_png, render_png, to_markdown, to_text
from .prompts import CASE_NAMES, CASES, PROMPT_SET_VERSION, get_cases, prompt_set_hash
from .vram import VramSampler

EXIT_OK, EXIT_USAGE, EXIT_SERVER = 0, 2, 3


def _err(msg: str) -> None:
    print(f"amagi-tokbench: {msg}", file=sys.stderr)


def _progress(quiet: bool):
    if quiet:
        return None
    return lambda msg: print(f"  {msg}", file=sys.stderr, flush=True)


def _write_outputs(result_or_results, prefix: str, png: bool, compare=False):
    prefix_path = Path(prefix)
    if prefix_path.parent and not prefix_path.parent.exists():
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    md = Path(f"{prefix}.md")
    md.write_text(
        compare_markdown(result_or_results) if compare else to_markdown(result_or_results), encoding="utf-8"
    )
    written.append(md)
    if png:
        try:
            if compare:
                written.append(render_compare_png(result_or_results, f"{prefix}.png"))
            else:
                written.append(render_png(result_or_results, f"{prefix}.png"))
        except ImportError:
            _err("Pillow is not installed, skipping the PNG (pip install Pillow)")
    return written


def _load(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchError(f"cannot read {path}: {exc}") from None
    if not isinstance(data, dict) or data.get("tool") != "amagi-tokbench" or "cases" not in data:
        raise BenchError(f"{path} is not an amagi-tokbench results file")
    return data


def cmd_run(args) -> int:
    try:
        cases = get_cases(args.cases.split(",") if args.cases else None)
    except ValueError as exc:
        _err(str(exc))
        return EXIT_USAGE
    api_key = args.api_key or os.environ.get("TOKBENCH_API_KEY")
    try:
        kind, url = find_server(args.backend, args.url, api_key=api_key)
    except BenchError as exc:
        _err(str(exc))
        return EXIT_SERVER
    backend = make_backend(kind, url, model=args.model, api_key=api_key, timeout=args.timeout,
                           num_ctx=args.num_ctx)
    sampler = None if args.no_vram else VramSampler(args.nvidia_smi)
    say = _progress(args.quiet)
    if say:
        say(f"{BACKEND_LABELS[kind]} at {backend.url}")
    try:
        result = run_benchmark(backend, cases, runs=args.runs, warmup=args.warmup, max_tokens=args.max_tokens,
                               label=args.label, sampler=sampler, progress=say)
    except BenchError as exc:
        _err(str(exc))
        return EXIT_SERVER
    except KeyboardInterrupt:
        _err("interrupted")
        return 130
    out = Path(f"{args.out}.json")
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    written = [out] + _write_outputs(result, args.out, png=not args.no_png)
    print()
    print(to_text(result), end="")
    print()
    print("wrote " + ", ".join(str(p) for p in written))
    return EXIT_OK


def cmd_card(args) -> int:
    try:
        result = _load(args.results)
    except BenchError as exc:
        _err(str(exc))
        return EXIT_USAGE
    prefix = args.out or str(Path(args.results).with_suffix(""))
    written = _write_outputs(result, prefix, png=not args.no_png)
    print(to_text(result), end="")
    print()
    print("wrote " + ", ".join(str(p) for p in written))
    return EXIT_OK


def cmd_compare(args) -> int:
    try:
        results = [_load(p) for p in args.results]
    except BenchError as exc:
        _err(str(exc))
        return EXIT_USAGE
    written = _write_outputs(results, args.out, png=not args.no_png, compare=True)
    print(compare_text(results), end="")
    print()
    print("wrote " + ", ".join(str(p) for p in written))
    return EXIT_OK


def cmd_detect(args) -> int:
    api_key = args.api_key or os.environ.get("TOKBENCH_API_KEY")
    try:
        kind, url = find_server(args.backend, args.url, api_key=api_key)
        backend = make_backend(kind, url, model=args.model, api_key=api_key)
        backend.resolve_model()
    except BenchError as exc:
        _err(str(exc))
        return EXIT_SERVER
    info = backend.describe()
    print(f"{info['name']} at {backend.url}")
    for key in ("version", "model", "model_file", "quant", "params", "context"):
        if info.get(key):
            print(f"  {key:<10} {info[key]}")
    return EXIT_OK


def cmd_prompts(args) -> int:
    print(f"prompt set {PROMPT_SET_VERSION}  sha256 {prompt_set_hash()[:16]}")
    for c in CASES:
        print(f"  {c.name:<7} {c.words:>5} words  max {c.max_tokens} new tokens  {c.description}")
    if args.show:
        for c in get_cases([args.show]):
            print()
            print(c.prompt)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="amagi-tokbench",
        description="Benchmark a local LLM server (Ollama, llama.cpp, vLLM, any OpenAI-compatible) with a "
                    "fixed prompt set and write a shareable markdown + PNG card.",
    )
    p.add_argument("--version", action="version", version=f"amagi-tokbench {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    def server_args(sp):
        sp.add_argument("--backend", choices=("auto",) + BACKENDS, default="auto",
                        help="server type (default: auto-detect)")
        sp.add_argument("--url", help="server address (default: try localhost ports 11434, 8080, 8000, 1234)")
        sp.add_argument("--model", "-m", help="model name (default: the one the server has loaded)")
        sp.add_argument("--api-key", help="bearer token, if the server wants one (or set TOKBENCH_API_KEY)")

    r = sub.add_parser("run", help="run the benchmark and write the card")
    server_args(r)
    r.add_argument("--runs", type=int, default=3, help="timed runs per case (default 3; the card shows the median)")
    r.add_argument("--warmup", type=int, default=1, help="untimed requests first, to load the model (default 1)")
    r.add_argument("--cases", help=f"comma list from {','.join(CASE_NAMES)} (default: all)")
    r.add_argument("--max-tokens", type=int, help="override new tokens per request for every case")
    r.add_argument("--label", help="free text for the card, e.g. 'RTX 3090 FE, 300 W cap'")
    r.add_argument("--out", "-o", default="tokbench", help="output prefix: writes PREFIX.json/.md/.png (default tokbench)")
    r.add_argument("--no-png", action="store_true", help="skip the PNG card")
    r.add_argument("--nvidia-smi", default="nvidia-smi", metavar="CMD",
                   help="command that runs nvidia-smi, e.g. 'ssh gpu-box nvidia-smi' for a remote server")
    r.add_argument("--no-vram", action="store_true", help="don't sample VRAM")
    r.add_argument("--num-ctx", type=int, help="Ollama only: context size to request (num_ctx)")
    r.add_argument("--timeout", type=float, default=300.0, help="seconds per request (default 300)")
    r.add_argument("--quiet", "-q", action="store_true", help="no progress lines")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("card", help="re-render the card from a results .json")
    c.add_argument("results")
    c.add_argument("--out", "-o", help="output prefix (default: next to the .json)")
    c.add_argument("--no-png", action="store_true")
    c.set_defaults(func=cmd_card)

    cp = sub.add_parser("compare", help="one table and PNG from several results files")
    cp.add_argument("results", nargs="+")
    cp.add_argument("--out", "-o", default="tokbench_compare", help="output prefix (default tokbench_compare)")
    cp.add_argument("--no-png", action="store_true")
    cp.set_defaults(func=cmd_compare)

    d = sub.add_parser("detect", help="show which server is listening and what it has loaded")
    server_args(d)
    d.set_defaults(func=cmd_detect)

    pr = sub.add_parser("prompts", help="list the fixed prompt set")
    pr.add_argument("--show", choices=CASE_NAMES, help="print one prompt in full")
    pr.set_defaults(func=cmd_prompts)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE
    if getattr(args, "runs", 1) < 1 or getattr(args, "warmup", 0) < 0:
        _err("--runs must be at least 1 and --warmup at least 0")
        return EXIT_USAGE
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
