import json
import socket
import subprocess
import sys

import pytest

from amagi_tokbench import __version__, backends
from amagi_tokbench.cli import main


def run_cli(*args):
    return main([str(a) for a in args])


@pytest.mark.parametrize("kind,name", [("llamacpp", "llama.cpp server"), ("vllm", "vLLM"),
                                       ("openai", "OpenAI-compatible server"), ("ollama", "Ollama")])
def test_run_end_to_end(fake_server, fake_smi_cmd, tmp_path, capsys, kind, name):
    srv = fake_server(kind)
    out = tmp_path / "res" / "card"
    code = run_cli("run", "--url", srv.url, "--runs", 2, "--warmup", 1, "--max-tokens", 40,
                   "--nvidia-smi", fake_smi_cmd, "--label", "fake rig", "-o", out)
    assert code == 0
    cap = capsys.readouterr()
    assert f"{name} at {srv.url}" in cap.err  # progress goes to stderr
    assert "vram: sampling" in cap.err
    assert "Generation" in cap.out and "wrote" in cap.out
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["backend"]["kind"] == kind
    assert [c["name"] for c in data["cases"]] == ["short", "medium", "long"]
    assert data["vram"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert data["vram"]["peak_mib"] >= 4000
    assert out.with_suffix(".md").read_text(encoding="utf-8").startswith("### ")
    assert out.with_suffix(".png").stat().st_size > 10_000


def test_run_backend_override_and_no_png(fake_server, tmp_path):
    srv = fake_server("llamacpp")
    out = tmp_path / "x"
    assert run_cli("run", "--backend", "openai", "--url", srv.url + "/v1", "--runs", 1, "--warmup", 0,
                   "--cases", "short", "--max-tokens", 40, "--no-png", "--no-vram", "-q", "-o", out) == 0
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["backend"]["kind"] == "openai" and data["vram"] is None
    assert not out.with_suffix(".png").exists()
    assert "cache_prompt" not in srv.requests[-1]  # plain OpenAI payload


def test_ollama_truncation_warning_and_num_ctx(fake_server, tmp_path, capsys):
    srv = fake_server("ollama", ctx_limit=2048)
    out = tmp_path / "o"
    assert run_cli("run", "--url", srv.url, "--runs", 1, "--warmup", 0, "--cases", "medium,long",
                   "--max-tokens", 40, "--no-vram", "--no-png", "-q", "-o", out) == 0
    assert "--num-ctx" in capsys.readouterr().out
    assert run_cli("run", "--url", srv.url, "--runs", 1, "--warmup", 0, "--cases", "medium,long",
                   "--max-tokens", 40, "--no-vram", "--no-png", "-q", "--num-ctx", 4096, "-o", out) == 0
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["warnings"] == [] and data["backend"]["context"] == 4096


def test_missing_nvidia_smi_does_not_stop_the_run(fake_server, tmp_path, capsys):
    srv = fake_server("vllm")
    out = tmp_path / "v"
    assert run_cli("run", "--url", srv.url, "--runs", 1, "--warmup", 0, "--cases", "short", "--max-tokens", 40,
                   "--nvidia-smi", "no-such-smi-tool", "--no-png", "-o", out) == 0
    assert "vram: off" in capsys.readouterr().err
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["vram"] is None


def test_card_and_compare_rerender(fake_server, tmp_path, capsys):
    a, b = fake_server("llamacpp"), fake_server("vllm")
    for srv, name in ((a, "a"), (b, "b")):
        assert run_cli("run", "--url", srv.url, "--runs", 1, "--warmup", 0, "--max-tokens", 40, "--no-vram",
                       "--no-png", "-q", "--label", name.upper(), "-o", tmp_path / name) == 0
    assert run_cli("card", tmp_path / "a.json") == 0
    assert (tmp_path / "a.png").exists()
    assert run_cli("compare", tmp_path / "a.json", tmp_path / "b.json", "-o", tmp_path / "cmp") == 0
    md = (tmp_path / "cmp.md").read_text(encoding="utf-8")
    assert "| A |" in md and "| B |" in md
    assert (tmp_path / "cmp.png").exists()
    capsys.readouterr()


def test_detect(fake_server, capsys):
    srv = fake_server("llamacpp")
    assert run_cli("detect", "--url", srv.url) == 0
    out = capsys.readouterr().out
    assert "llama.cpp server" in out and "Q4_K_M" in out and "b6000-abc1234" in out


def test_auto_detect_scans_default_ports(fake_server, monkeypatch):
    srv = fake_server("ollama")
    for k in backends.BACKENDS:
        monkeypatch.setitem(backends.DEFAULT_URLS, k, "http://127.0.0.1:9" if k != "ollama" else srv.url)
    assert backends.find_server() == ("ollama", srv.url)


def _closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_errors_and_exit_codes(fake_server, tmp_path, capsys):
    assert run_cli("run", "--url", f"http://127.0.0.1:{_closed_port()}", "--no-vram") == 3
    assert "nothing answered" in capsys.readouterr().err
    srv = fake_server("vllm", fail_chat=True)
    assert run_cli("run", "--url", srv.url, "--no-vram", "--warmup", 0, "-o", tmp_path / "f") == 3
    assert "model not loaded" in capsys.readouterr().err
    assert run_cli("run", "--runs", 0) == 2
    assert run_cli("run", "--cases", "huge") == 2
    assert run_cli("card", tmp_path / "missing.json") == 2
    (tmp_path / "other.json").write_text('{"hello": 1}', encoding="utf-8")
    assert run_cli("card", tmp_path / "other.json") == 2
    assert "not an amagi-tokbench results file" in capsys.readouterr().err
    assert run_cli() == 2


def test_prompts_command(capsys):
    assert run_cli("prompts", "--show", "short") == 0
    out = capsys.readouterr().out
    assert "prompt set v1" in out and "long" in out and "words" in out


def test_console_entry_point():
    r = subprocess.run([sys.executable, "-m", "amagi_tokbench", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == f"amagi-tokbench {__version__}"
