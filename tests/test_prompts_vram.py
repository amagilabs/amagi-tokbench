import os
import time

import pytest

from amagi_tokbench.prompts import CASE_NAMES, CASES, get_cases, prompt_set_hash, with_nonce
from amagi_tokbench.vram import VramSampler, parse_line, public_source, split_command


def test_prompt_set_is_pinned():
    # Changing any prompt changes every number ever published with v1. Bump the version instead.
    assert prompt_set_hash().startswith("f8187fc0240defb6")
    assert [c.words for c in CASES] == [23, 399, 1607]
    assert CASE_NAMES == ("short", "medium", "long")


def test_get_cases():
    assert [c.name for c in get_cases(None)] == ["short", "medium", "long"]
    assert [c.name for c in get_cases(["long", " short"])] == ["short", "long"]  # fixed order
    with pytest.raises(ValueError, match="unknown case"):
        get_cases(["huge"])


def test_nonce_goes_first_so_prefix_caches_miss():
    a, b = with_nonce("same prompt", "aa"), with_nonce("same prompt", "bb")
    assert a.endswith("same prompt") and a[:8] != b[:8]


def test_parse_line():
    assert parse_line("0, NVIDIA GeForce RTX 3090, 5821, 24576") == {
        "index": 0, "name": "NVIDIA GeForce RTX 3090", "used_mib": 5821.0, "total_mib": 24576.0}
    assert parse_line("1, Odd, Name, 100, 200")["name"] == "Odd, Name"
    assert parse_line("NVIDIA-SMI has failed") is None
    assert parse_line("0, GPU, [N/A], 24576") is None


def test_feed_tracks_peak_per_gpu():
    s = VramSampler(None)
    for line in ["0, A, 1000, 24576", "1, A, 500, 24576", "0, A, 3000, 24576", "0, A, 2000, 24576", "junk"]:
        s.feed(line)
    r = s.result()
    assert r["samples"] == 4
    assert [g["peak_mib"] for g in r["gpus"]] == [3000, 500]
    assert [g["start_mib"] for g in r["gpus"]] == [1000, 500]
    assert r["peak_mib"] == 3500 and r["total_mib"] == 49152


def test_missing_command_is_reported_not_raised():
    s = VramSampler("definitely-not-nvidia-smi-xyz")
    assert s.start() is False
    assert "not found" in s.error
    assert s.result() is None


def test_sampler_reads_a_live_process(fake_smi_cmd):
    s = VramSampler(fake_smi_cmd, interval_ms=20)
    assert s.start(), s.error
    time.sleep(0.4)
    s.stop()
    r = s.result()
    assert r["samples"] >= 3
    assert r["gpus"][0]["start_mib"] == 4000
    assert r["peak_mib"] > 4000


@pytest.mark.skipif(os.name != "nt", reason="Windows path handling")
def test_split_command_keeps_windows_paths():
    assert split_command(r'"C:\Program Files\NVIDIA\nvidia-smi.exe"') == [r"C:\Program Files\NVIDIA\nvidia-smi.exe"]
    assert split_command(r"C:\Tools\nvidia-smi.exe") == [r"C:\Tools\nvidia-smi.exe"]


def test_split_command_remote():
    assert split_command("ssh gpu-box nvidia-smi") == ["ssh", "gpu-box", "nvidia-smi"]


def test_results_never_carry_the_smi_command():
    assert public_source("nvidia-smi") == "nvidia-smi"
    assert public_source("/usr/bin/nvidia-smi") == "nvidia-smi"
    assert public_source("ssh admin@gpu-box.lan nvidia-smi") == "nvidia-smi (custom command)"
