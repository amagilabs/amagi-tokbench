"""Talk to the servers: detect which backend is listening, describe it, stream one request.

Stdlib only (urllib). The stream parsers are plain functions over (timestamp, line) pairs so
the timing math can be tested without a server.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, List, Optional, Tuple

BACKENDS = ("ollama", "llamacpp", "vllm", "openai")
BACKEND_LABELS = {
    "ollama": "Ollama",
    "llamacpp": "llama.cpp server",
    "vllm": "vLLM",
    "openai": "OpenAI-compatible server",
}
DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "llamacpp": "http://localhost:8080",
    "vllm": "http://localhost:8000",
    "openai": "http://localhost:1234",
}
SEED = 42


class BenchError(Exception):
    """A server could not be reached or answered with an error."""


# --------------------------------------------------------------------------- HTTP


def _request(url, payload=None, api_key=None, timeout=10.0):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:  # pragma: no cover - best effort only
        body = ""
    try:
        obj = json.loads(body)
        err = obj.get("error", obj)
        if isinstance(err, dict):
            err = err.get("message", err)
        body = str(err)
    except ValueError:
        pass
    return f"HTTP {exc.code}" + (f": {body[:300]}" if body else "")


def get_json(url, api_key=None, timeout=5.0, payload=None):
    """GET (or POST when payload is given) a JSON document. Raises BenchError."""
    req = _request(url, payload=payload, api_key=api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise BenchError(f"{url}: {_error_text(exc)}") from None
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise BenchError(f"{url}: {reason}") from None
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        raise BenchError(f"{url}: answer is not JSON") from None


def stream_lines(url, payload, api_key=None, timeout=300.0) -> Iterator[Tuple[float, bytes]]:
    """POST and yield (perf_counter time, line) for every line of the streamed answer."""
    req = _request(url, payload=payload, api_key=api_key)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise BenchError(f"{url}: {_error_text(exc)}") from None
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise BenchError(f"{url}: {reason}") from None
    with resp:
        while True:
            try:
                line = resp.readline()
            except OSError as exc:
                raise BenchError(f"{url}: stream broke: {exc}") from None
            if not line:
                return
            yield time.perf_counter(), line


def normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    for suffix in ("/v1/chat/completions", "/api/chat", "/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


# --------------------------------------------------------------------------- one request


@dataclass
class Sample:
    """What one streamed request measured. Times are seconds; None means not reported."""

    ttft_s: float
    gen_s: float
    total_s: float
    gen_tokens: int
    prompt_tokens: Optional[int] = None
    token_source: str = "chunks"  # where gen_tokens came from: usage, timings, server, chunks
    server_pp_tps: Optional[float] = None
    server_gen_tps: Optional[float] = None
    finish_reason: Optional[str] = None
    text_chars: int = 0

    @property
    def pp_tps(self) -> Optional[float]:
        """Prompt tokens per second, client-side: prompt tokens / time to first token."""
        if not self.prompt_tokens or self.ttft_s <= 0:
            return None
        return self.prompt_tokens / self.ttft_s

    @property
    def gen_tps(self) -> Optional[float]:
        """Generation tokens per second, client-side: tokens after the first / time they took."""
        if self.gen_tokens < 2 or self.gen_s <= 0:
            return None
        return (self.gen_tokens - 1) / self.gen_s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pp_tps"] = self.pp_tps
        d["gen_tps"] = self.gen_tps
        return d


@dataclass
class _Acc:
    t0: float
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    chunks: int = 0
    chars: int = 0
    extra: dict = field(default_factory=dict)

    def text(self, t: float, s: str) -> None:
        if not s:
            return
        if self.t_first is None:
            self.t_first = t
        self.t_last = t
        self.chunks += 1
        self.chars += len(s)

    def sample(self, t_end, gen_tokens, token_source, **kw) -> Sample:
        if self.t_first is None:
            raise BenchError("the server streamed no text (check the model name and the server log)")
        return Sample(
            ttft_s=self.t_first - self.t0,
            gen_s=(self.t_last or self.t_first) - self.t_first,
            total_s=t_end - self.t0,
            gen_tokens=gen_tokens,
            token_source=token_source,
            text_chars=self.chars,
            **kw,
        )


def parse_openai_stream(t0: float, lines: Iterable[Tuple[float, bytes]]) -> Sample:
    """Parse an OpenAI-style SSE chat stream (llama.cpp, vLLM, LM Studio, ...)."""
    acc = _Acc(t0)
    usage = None
    timings = None
    finish = None
    t_end = t0
    for t, raw in lines:
        t_end = t
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("error"):
            err = obj["error"]
            raise BenchError(str(err.get("message", err) if isinstance(err, dict) else err))
        for ch in obj.get("choices") or []:
            delta = ch.get("delta") or {}
            piece = (delta.get("reasoning_content") or delta.get("reasoning") or "") + (
                delta.get("content") or ""
            )
            acc.text(t, piece)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
        if obj.get("usage"):
            usage = obj["usage"]
        if obj.get("timings"):
            timings = obj["timings"]

    prompt_tokens = None
    gen_tokens, source = acc.chunks, "chunks"
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        if usage.get("completion_tokens"):
            gen_tokens, source = int(usage["completion_tokens"]), "usage"
    server_pp = server_gen = None
    if timings:
        if prompt_tokens is None and timings.get("prompt_n") is not None:
            prompt_tokens = int(timings["prompt_n"]) + int(timings.get("cache_n") or 0)
        if source == "chunks" and timings.get("predicted_n"):
            gen_tokens, source = int(timings["predicted_n"]), "timings"
        server_pp = _rate(timings.get("prompt_n"), timings.get("prompt_ms"), 1e3)
        server_gen = _rate(timings.get("predicted_n"), timings.get("predicted_ms"), 1e3)
    return acc.sample(
        t_end,
        gen_tokens,
        source,
        prompt_tokens=prompt_tokens,
        server_pp_tps=server_pp,
        server_gen_tps=server_gen,
        finish_reason=finish,
    )


def parse_ollama_stream(t0: float, lines: Iterable[Tuple[float, bytes]]) -> Sample:
    """Parse Ollama's NDJSON /api/chat (or /api/generate) stream."""
    acc = _Acc(t0)
    final = None
    t_end = t0
    for t, raw in lines:
        t_end = t
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if obj.get("error"):
            raise BenchError(str(obj["error"]))
        msg = obj.get("message") or {}
        acc.text(t, (msg.get("thinking") or "") + (msg.get("content") or "") + (obj.get("response") or ""))
        if obj.get("done"):
            final = obj
            break
    final = final or {}
    gen_tokens, source = acc.chunks, "chunks"
    if final.get("eval_count"):
        gen_tokens, source = int(final["eval_count"]), "server"
    return acc.sample(
        t_end,
        gen_tokens,
        source,
        prompt_tokens=final.get("prompt_eval_count"),
        server_pp_tps=_rate(final.get("prompt_eval_count"), final.get("prompt_eval_duration"), 1e9),
        server_gen_tps=_rate(final.get("eval_count"), final.get("eval_duration"), 1e9),
        finish_reason=final.get("done_reason"),
    )


def _rate(n, duration, per_second) -> Optional[float]:
    try:
        n, duration = float(n), float(duration)
    except (TypeError, ValueError):
        return None
    if n <= 0 or duration <= 0:
        return None
    return n / (duration / per_second)


# --------------------------------------------------------------------------- backends

_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])(I?Q\d+_[A-Z0-9_]+?|Q\d+_\d|F16|BF16|F32|MXFP4|FP8|AWQ|GPTQ|INT8|INT4)(?=[.\-_]|$)",
    re.IGNORECASE,
)


def quant_from_name(name: str) -> Optional[str]:
    """Pull a quant tag (Q4_K_M, IQ3_XXS, BF16, AWQ, ...) out of a model file or repo name."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"\.gguf$", "", base, flags=re.IGNORECASE)
    hits = _QUANT_RE.findall(base)
    return hits[-1].upper() if hits else None


class Backend:
    kind = "openai"

    def __init__(self, url, model=None, api_key=None, timeout=300.0, num_ctx=None):
        self.url = normalize_url(url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.num_ctx = num_ctx

    @property
    def label(self) -> str:
        return BACKEND_LABELS[self.kind]

    def _get(self, path, payload=None, timeout=10.0):
        return get_json(self.url + path, api_key=self.api_key, timeout=timeout, payload=payload)

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        data = self._get("/v1/models").get("data") or []
        if not data:
            raise BenchError(f"{self.url}/v1/models lists no models; pass --model")
        self.model = data[0]["id"]
        return self.model

    def describe(self) -> dict:
        """Server facts for the card. Never raises: missing facts are just left out."""
        info = {"kind": self.kind, "name": self.label, "url": self.url, "model": self.model}
        try:
            models = self._get("/v1/models").get("data") or []
            for m in models:
                if m.get("id") == self.model:
                    if m.get("max_model_len"):
                        info["context"] = m["max_model_len"]
                    if m.get("owned_by"):
                        info["owned_by"] = m["owned_by"]
        except BenchError:
            pass
        quant = quant_from_name(self.model or "")
        if quant:
            info["quant"] = quant
        return info

    def payload(self, prompt: str, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": SEED,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    def generate(self, prompt: str, max_tokens: int) -> Sample:
        url = self.url + "/v1/chat/completions"
        t0 = time.perf_counter()
        return parse_openai_stream(t0, stream_lines(url, self.payload(prompt, max_tokens), self.api_key, self.timeout))


class LlamaCppBackend(Backend):
    kind = "llamacpp"

    def payload(self, prompt, max_tokens):
        p = super().payload(prompt, max_tokens)
        p["cache_prompt"] = False
        p["ignore_eos"] = True
        return p

    def describe(self) -> dict:
        info = super().describe()
        try:
            props = self._get("/props")
        except BenchError:
            return info
        if props.get("build_info"):
            info["version"] = str(props["build_info"])
        path = props.get("model_path") or ""
        if path:
            info["model_file"] = re.split(r"[\\/]", path)[-1]
            quant = quant_from_name(path)
            if quant:
                info["quant"] = quant
        gen = props.get("default_generation_settings") or {}
        n_ctx = gen.get("n_ctx") or props.get("n_ctx")
        if n_ctx:
            info["context"] = n_ctx
        if props.get("total_slots"):
            info["slots"] = props["total_slots"]
        return info


class VllmBackend(Backend):
    kind = "vllm"

    def payload(self, prompt, max_tokens):
        p = super().payload(prompt, max_tokens)
        p["ignore_eos"] = True
        return p

    def describe(self) -> dict:
        info = super().describe()
        try:
            v = self._get("/version")
            if v.get("version"):
                info["version"] = str(v["version"])
        except BenchError:
            pass
        return info


class OllamaBackend(Backend):
    kind = "ollama"

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        running = [m.get("name") or m.get("model") for m in (self._safe("/api/ps").get("models") or [])]
        if len(running) == 1:
            self.model = running[0]
            return self.model
        tags = [m.get("name") or m.get("model") for m in (self._get("/api/tags").get("models") or [])]
        if len(tags) == 1:
            self.model = tags[0]
            return self.model
        if not tags:
            raise BenchError("Ollama has no models pulled; run `ollama pull <model>` first")
        raise BenchError("pass --model, one of: " + ", ".join(sorted(t for t in tags if t)))

    def _safe(self, path, payload=None) -> dict:
        try:
            return self._get(path, payload=payload)
        except BenchError:
            return {}

    def describe(self) -> dict:
        info = {"kind": self.kind, "name": self.label, "url": self.url, "model": self.model}
        v = self._safe("/api/version")
        if v.get("version"):
            info["version"] = str(v["version"])
        show = self._safe("/api/show", payload={"model": self.model})
        details = show.get("details") or {}
        if details.get("quantization_level"):
            info["quant"] = details["quantization_level"]
        if details.get("parameter_size"):
            info["params"] = details["parameter_size"]
        if details.get("family"):
            info["family"] = details["family"]
        if self.num_ctx:
            info["context"] = self.num_ctx
        for m in self._safe("/api/ps").get("models") or []:
            if (m.get("name") or m.get("model")) == self.model:
                if m.get("size_vram"):
                    info["ollama_size_vram_mib"] = round(m["size_vram"] / 2**20)
                if m.get("context_length"):
                    info["context"] = m["context_length"]
        return info

    def payload(self, prompt, max_tokens):
        options = {"temperature": 0, "seed": SEED, "num_predict": max_tokens}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": options,
            "keep_alive": "10m",
        }

    def generate(self, prompt, max_tokens):
        url = self.url + "/api/chat"
        t0 = time.perf_counter()
        return parse_ollama_stream(t0, stream_lines(url, self.payload(prompt, max_tokens), self.api_key, self.timeout))


_CLASSES = {"ollama": OllamaBackend, "llamacpp": LlamaCppBackend, "vllm": VllmBackend, "openai": Backend}


def make_backend(kind, url, **kw) -> Backend:
    return _CLASSES[kind](url, **kw)


def detect_kind(url, api_key=None, timeout=3.0) -> Optional[str]:
    """Work out which server is listening at url. None when nothing answers."""
    base = normalize_url(url)

    def probe(path):
        try:
            return get_json(base + path, api_key=api_key, timeout=timeout)
        except BenchError:
            return None

    v = probe("/api/version")
    if isinstance(v, dict) and "version" in v:
        return "ollama"
    props = probe("/props")
    if isinstance(props, dict) and ("default_generation_settings" in props or "build_info" in props):
        return "llamacpp"
    models = probe("/v1/models")
    owners = {m.get("owned_by") for m in (models or {}).get("data") or [] if isinstance(m, dict)}
    if "vllm" in owners:
        return "vllm"
    version = probe("/version")
    if isinstance(version, dict) and "version" in version and models is not None:
        return "vllm"
    if models is not None:
        return "openai"
    return None


def find_server(kind=None, url=None, api_key=None) -> Tuple[str, str]:
    """Return (kind, url). With no url, try each backend's default port on localhost."""
    if url:
        found = detect_kind(url, api_key=api_key)
        if kind and kind != "auto":
            if found is None:
                raise BenchError(f"nothing answered at {normalize_url(url)}")
            return kind, url
        if found is None:
            raise BenchError(f"nothing answered at {normalize_url(url)} (is the server running?)")
        return found, url
    candidates: List[Tuple[str, str]]
    if kind and kind != "auto":
        candidates = [(kind, DEFAULT_URLS[kind])]
    else:
        candidates = [(k, DEFAULT_URLS[k]) for k in BACKENDS]
    for want, candidate in candidates:
        found = detect_kind(candidate, api_key=api_key, timeout=1.5)
        if found:
            return (kind if kind and kind != "auto" else found), candidate
    tried = ", ".join(u for _, u in candidates)
    raise BenchError(f"no server found (tried {tried}); start one or pass --url")
