import pytest

from amagi_tokbench.backends import make_backend
from amagi_tokbench.bench import check_warnings, headline, run_benchmark
from amagi_tokbench.card import (
    compare_markdown,
    compare_text,
    model_text,
    render_compare_png,
    render_png,
    subtitle,
    to_markdown,
    to_text,
)
from amagi_tokbench.prompts import get_cases


def case(name, prompt_tokens, gen_tokens=128, gen_tps=100.0, pp_tps=2000.0, ttft_ms=50.0):
    return {"name": name, "median": {"prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens, "ttft_ms": ttft_ms,
                                     "pp_tps": pp_tps, "gen_tps": gen_tps, "server_pp_tps": None,
                                     "server_gen_tps": None}}


def test_headline_picks_short_for_generation_and_longest_prompt_for_pp():
    h = headline([case("short", 70, gen_tps=140.0, ttft_ms=40.0), case("medium", 640, pp_tps=3500.0),
                  case("long", 2500, pp_tps=4500.0, gen_tps=130.0)])
    assert (h["gen_tps"], h["ttft_ms"], h["gen_case"]) == (140.0, 40.0, "short")
    assert (h["pp_tps"], h["pp_case"]) == (4500.0, "long")
    h = headline([case("long", 2500, gen_tps=130.0)])
    assert (h["gen_case"], h["pp_case"]) == ("long", "long")
    assert headline([])["gen_tps"] is None


def test_warnings():
    assert check_warnings([case("short", 70), case("medium", 640), case("long", 2500)]) == []
    w = check_warnings([case("short", 70, gen_tokens=9)])
    assert len(w) == 1 and "stopped after 9 tokens" in w[0]
    w = check_warnings([case("short", None)])
    assert "prompt speed is unknown" in w[0]
    # Ollama's default context cut the long prompt to 2,048 tokens
    w = check_warnings([case("medium", 640), case("long", 2048)])
    assert len(w) == 1 and "--num-ctx" in w[0]


@pytest.mark.parametrize("kind", ["llamacpp", "vllm", "openai", "ollama"])
def test_run_benchmark_against_fake_server(fake_server, kind):
    srv = fake_server(kind)
    backend = make_backend(kind, srv.url)
    messages = []
    result = run_benchmark(backend, get_cases(["short", "medium"]), runs=2, warmup=1, max_tokens=40,
                           label="test rig", progress=messages.append)
    assert result["tool"] == "amagi-tokbench" and result["label"] == "test rig"
    assert [c["name"] for c in result["cases"]] == ["short", "medium"]
    assert len(srv.requests) == 1 + 2 * 2  # warmup + 2 runs x 2 cases
    # every timed prompt carries a fresh nonce so no prefix cache can serve it
    starts = [r["messages"][0]["content"].split("\n", 1)[0] for r in srv.requests]
    assert len(set(starts)) == len(starts)
    for c in result["cases"]:
        assert c["median"]["gen_tokens"] == 40
        assert c["median"]["gen_tps"] > 0 and c["median"]["ttft_ms"] > 0
        assert c["median"]["prompt_tokens"] > 0
    assert result["summary"]["gen_case"] == "short" and result["summary"]["pp_case"] == "medium"
    assert result["settings"]["temperature"] == 0 and result["settings"]["runs"] == 2
    assert any("warmup" in m for m in messages)


def test_backend_specific_payloads(fake_server):
    srv = fake_server("llamacpp")
    make_backend("llamacpp", srv.url, model="m").generate("hi", 8)
    p = srv.requests[-1]
    assert p["cache_prompt"] is False and p["ignore_eos"] is True and p["temperature"] == 0
    assert p["stream_options"] == {"include_usage": True}

    srv = fake_server("ollama")
    make_backend("ollama", srv.url, model="qwen2.5:7b", num_ctx=4096).generate("hi", 8)
    p = srv.requests[-1]
    assert p["options"] == {"temperature": 0, "seed": 42, "num_predict": 8, "num_ctx": 4096}


@pytest.mark.parametrize("kind,expect", [
    ("llamacpp", {"version": "b6000-abc1234", "model_file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "quant": "Q4_K_M",
                  "context": 8192, "slots": 1}),
    ("vllm", {"version": "0.11.0", "quant": "AWQ", "context": 8192}),
    ("ollama", {"version": "0.12.6", "quant": "Q4_K_M", "params": "7.6B", "context": 4096,
                "ollama_size_vram_mib": 5120}),
])
def test_describe(fake_server, kind, expect):
    srv = fake_server(kind)
    b = make_backend(kind, srv.url)
    b.resolve_model()
    info = b.describe()
    for k, v in expect.items():
        assert info[k] == v, k


def _result(model_file=None, model="m", gpu=True, label=None, warnings=()):
    vram = {"gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "total_mib": 24576, "start_mib": 900,
                      "peak_mib": 5120}], "peak_mib": 5120, "total_mib": 24576, "samples": 40,
            "source": "nvidia-smi"} if gpu else None
    cases = [case("short", 71, 256, 142.5, 1829.0, 39.0), case("medium", 637, 128, 142.0, 3543.0, 180.0),
             case("long", 2472, 128, 137.1, 4497.0, 550.0)]
    return {
        "tool": "amagi-tokbench", "tool_version": "0.1.0", "created": "2026-10-24T15:00:00Z", "label": label,
        "backend": {"kind": "llamacpp", "name": "llama.cpp server", "version": "b6000", "model": model,
                    "model_file": model_file, "quant": "Q4_K_M"},
        "vram": vram, "settings": {"prompt_set": "v1", "runs": 3, "warmup": 1},
        "cases": cases, "summary": headline(cases), "warnings": list(warnings),
    }


def test_markdown_and_text_cards():
    r = _result(model_file="Qwen2.5-7B-Instruct-Q4_K_M.gguf", label="RTX 3090 FE", warnings=["long: cut"])
    assert model_text(r) == "Qwen2.5-7B-Instruct-Q4_K_M"
    assert subtitle(r) == "llama.cpp server b6000 · GeForce RTX 3090"  # quant already in the name
    md = to_markdown(r)
    assert md.startswith("### Qwen2.5-7B-Instruct-Q4_K_M")
    assert "**142.5 tok/s** generation" in md and "**4,497 tok/s** prompt processing" in md
    assert "**5.0 GiB** peak vram" in md
    assert "| long | 2,472 | 128 | 550 ms | 4,497 | 137.1 |" in md
    assert "Note: long: cut" in md
    txt = to_text(r)
    assert txt.isascii()  # Windows consoles choke on anything else
    assert "RTX 3090 FE" in txt and "142.5" in txt


def test_card_without_vram_or_model_file():
    r = _result(model="qwen2.5:7b", gpu=False)
    assert model_text(r) == "qwen2.5:7b"
    assert "Q4_K_M" in subtitle(r)
    assert "**n/a** peak vram" in to_markdown(r)


def test_png_cards(tmp_path):
    from PIL import Image

    p = render_png(_result(model_file="x-Q4_K_M.gguf"), tmp_path / "card.png")
    with Image.open(p) as im:
        assert im.size == (1200, 675)
    p = render_png(_result(model_file="x-Q4_K_M.gguf", label="a very long label " * 10), tmp_path / "label.png")
    with Image.open(p) as im:
        assert im.size[0] == 1200 and im.size[1] > 675  # the label line adds height
    p = render_png(_result(gpu=False), tmp_path / "nogpu.png")
    assert p.stat().st_size > 10_000
    results = [_result(model="a", label="llama.cpp"), _result(model="b", label="vLLM", gpu=False)]
    p = render_compare_png(results, tmp_path / "cmp.png")
    with Image.open(p) as im:
        assert im.size[0] == 1200
    md = compare_markdown(results)
    assert "| llama.cpp |" in md and "| vLLM |" in md
    assert compare_text(results).isascii()
