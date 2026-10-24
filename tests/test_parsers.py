import json

import pytest

from amagi_tokbench.backends import (
    BenchError,
    Sample,
    normalize_url,
    parse_ollama_stream,
    parse_openai_stream,
    quant_from_name,
)


def sse(t, obj):
    return t, f"data: {json.dumps(obj)}\n".encode()


def delta(text, finish=None):
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


def test_openai_stream_timing_from_chunk_arrival():
    # first text at t=0.25 after t0=0; 10 more chunks every 0.1 s -> last at 1.25
    lines = [(0.1, b": keep-alive\n"), sse(0.2, {"choices": [{"delta": {"role": "assistant"}}]})]
    lines += [sse(0.25 + 0.1 * i, delta(f" t{i}")) for i in range(11)]
    lines += [sse(1.3, {"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 11}})]
    lines += [(1.31, b"data: [DONE]\n")]
    s = parse_openai_stream(0.0, lines)
    assert s.ttft_s == pytest.approx(0.25)
    assert s.gen_s == pytest.approx(1.0)
    assert s.total_s == pytest.approx(1.31)
    assert (s.gen_tokens, s.token_source, s.prompt_tokens) == (11, "usage", 500)
    assert s.gen_tps == pytest.approx(10.0)  # 10 tokens after the first, in 1.0 s
    assert s.pp_tps == pytest.approx(2000.0)  # 500 tokens / 0.25 s
    assert s.server_gen_tps is None


def test_openai_stream_llamacpp_timings_and_cache():
    lines = [sse(0.1, delta("a")), sse(0.2, delta("b", "length"))]
    lines += [sse(0.21, {"choices": [], "timings": {"prompt_n": 90, "cache_n": 10, "prompt_ms": 45.0,
                                                   "predicted_n": 2, "predicted_ms": 20.0}})]
    s = parse_openai_stream(0.0, lines)
    assert s.prompt_tokens == 100  # evaluated + cached
    assert s.gen_tokens == 2 and s.token_source == "timings"
    assert s.server_pp_tps == pytest.approx(2000.0)
    assert s.server_gen_tps == pytest.approx(100.0)
    assert s.finish_reason == "length"


def test_openai_stream_counts_reasoning_and_falls_back_to_chunks():
    lines = [sse(0.5, {"choices": [{"delta": {"reasoning_content": "hmm"}}]}), sse(0.6, delta("ok"))]
    s = parse_openai_stream(0.0, lines)
    assert s.ttft_s == pytest.approx(0.5)  # reasoning text is the first token
    assert (s.gen_tokens, s.token_source, s.prompt_tokens) == (2, "chunks", None)
    assert s.pp_tps is None


def test_openai_stream_error_and_empty():
    with pytest.raises(BenchError, match="context length"):
        parse_openai_stream(0.0, [sse(0.1, {"error": {"message": "context length exceeded"}})])
    with pytest.raises(BenchError, match="no text"):
        parse_openai_stream(0.0, [sse(0.1, {"choices": [{"delta": {"content": ""}}]}), (0.2, b"data: [DONE]\n")])


def test_ollama_stream():
    lines = [(0.3 + 0.05 * i, (json.dumps({"message": {"content": f"w{i}"}, "done": False}) + "\n").encode())
             for i in range(5)]
    lines.append((0.55, (json.dumps({"message": {"content": ""}, "done": True, "done_reason": "length",
                                     "prompt_eval_count": 600, "prompt_eval_duration": 200_000_000,
                                     "eval_count": 5, "eval_duration": 50_000_000}) + "\n").encode()))
    s = parse_ollama_stream(0.0, lines)
    assert s.ttft_s == pytest.approx(0.3)
    assert s.gen_s == pytest.approx(0.2)
    assert (s.gen_tokens, s.token_source, s.prompt_tokens) == (5, "server", 600)
    assert s.gen_tps == pytest.approx(20.0)
    assert s.server_pp_tps == pytest.approx(3000.0)
    assert s.server_gen_tps == pytest.approx(100.0)
    assert s.finish_reason == "length"


def test_ollama_stream_error():
    with pytest.raises(BenchError, match="not found"):
        parse_ollama_stream(0.0, [(0.1, b'{"error": "model \\"x\\" not found"}\n')])


def test_sample_rates_guard_small_values():
    assert Sample(ttft_s=0.0, gen_s=0.0, total_s=0.0, gen_tokens=1, prompt_tokens=10).pp_tps is None
    assert Sample(ttft_s=0.1, gen_s=0.0, total_s=0.1, gen_tokens=1).gen_tps is None
    d = Sample(ttft_s=0.1, gen_s=1.0, total_s=1.1, gen_tokens=3, prompt_tokens=50).to_dict()
    assert d["gen_tps"] == pytest.approx(2.0) and d["pp_tps"] == pytest.approx(500.0)


@pytest.mark.parametrize("name,quant", [
    ("/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf", "Q4_K_M"),
    ("gpt-oss-20b-MXFP4.gguf", "MXFP4"),
    ("Qwen3-30B-A3B-UD-Q4_K_XL.gguf", "Q4_K_XL"),
    ("llama-3.1-8b-instruct-q8_0.gguf", "Q8_0"),
    ("C:\\m\\Mistral-7B-IQ3_XXS.gguf", "IQ3_XXS"),
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", "AWQ"),
    ("model-bf16.gguf", "BF16"),
    ("qwen2.5:7b", None),
    ("", None),
])
def test_quant_from_name(name, quant):
    assert quant_from_name(name) == quant


@pytest.mark.parametrize("raw,clean", [
    ("localhost:8080", "http://localhost:8080"),
    ("http://h:8000/v1/", "http://h:8000"),
    ("http://h:11434/api/chat", "http://h:11434"),
    ("https://h/v1/chat/completions", "https://h"),
])
def test_normalize_url(raw, clean):
    assert normalize_url(raw) == clean
