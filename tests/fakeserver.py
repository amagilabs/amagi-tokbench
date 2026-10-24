"""A small in-process stand-in for each server type, so the CLI can be run end to end in tests.

It speaks just enough of each API: the detection endpoints, the model listing and a streamed
chat reply with a short sleep between chunks, so the client-side timers have something to time.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def prompt_tokens_for(text: str) -> int:
    return int(len(text.split()) * 1.5) + 12


class FakeServer:
    def __init__(self, kind: str, max_gen: int = 40, delay: float = 0.002, ctx_limit=None, fail_chat=False):
        self.kind = kind
        self.max_gen = max_gen
        self.delay = delay
        self.ctx_limit = ctx_limit  # ollama: truncate prompts when the client sends no num_ctx
        self.fail_chat = fail_chat
        self.requests = []  # chat payloads, in order
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                obj = server.get(self.path)
                if obj is None:
                    self._json({"error": "not found"}, 404)
                else:
                    self._json(obj)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/show" and server.kind == "ollama":
                    self._json({"details": {"quantization_level": "Q4_K_M", "parameter_size": "7.6B",
                                            "family": "qwen2"}})
                    return
                chat = "/api/chat" if server.kind == "ollama" else "/v1/chat/completions"
                if self.path != chat:
                    self._json({"error": "not found"}, 404)
                    return
                if server.fail_chat:
                    self._json({"error": {"message": "model not loaded"}}, 500)
                    return
                server.requests.append(payload)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in server.stream(payload):
                    self.wfile.write(line)
                    self.wfile.flush()
                    time.sleep(server.delay)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    # --------------------------------------------------------------- endpoints

    def get(self, path):
        k = self.kind
        if k == "ollama":
            return {
                "/api/version": {"version": "0.12.6"},
                "/api/tags": {"models": [{"name": "qwen2.5:7b"}]},
                "/api/ps": {"models": [{"name": "qwen2.5:7b", "size_vram": 5 * 2**30, "context_length": 4096}]},
            }.get(path)
        if k == "llamacpp":
            return {
                "/props": {"build_info": "b6000-abc1234", "model_path": "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                           "default_generation_settings": {"n_ctx": 8192}, "total_slots": 1},
                "/v1/models": {"data": [{"id": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "owned_by": "llamacpp"}]},
            }.get(path)
        if k == "vllm":
            return {
                "/v1/models": {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct-AWQ", "owned_by": "vllm",
                                         "max_model_len": 8192}]},
                "/version": {"version": "0.11.0"},
            }.get(path)
        return {"/v1/models": {"data": [{"id": "local-model", "owned_by": "organization_owner"}]}}.get(path)

    def stream(self, payload):
        prompt = payload["messages"][0]["content"]
        n_prompt = prompt_tokens_for(prompt)
        if self.kind == "ollama":
            opts = payload.get("options") or {}
            if self.ctx_limit and not opts.get("num_ctx"):
                n_prompt = min(n_prompt, self.ctx_limit)
            n_gen = min(opts.get("num_predict", 64), self.max_gen)
            for i in range(n_gen):
                yield (json.dumps({"message": {"role": "assistant", "content": f" w{i}"}, "done": False}) + "\n").encode()
            yield (json.dumps({
                "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "length",
                "prompt_eval_count": n_prompt, "prompt_eval_duration": 50_000_000,
                "eval_count": n_gen, "eval_duration": 400_000_000,
            }) + "\n").encode()
            return
        n_gen = min(payload.get("max_tokens", 64), self.max_gen)
        if payload.get("ignore_eos"):
            n_gen = payload.get("max_tokens", 64)
        for i in range(n_gen):
            chunk = {"choices": [{"index": 0, "delta": {"content": f" w{i}"}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        last = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
        yield f"data: {json.dumps(last)}\n\n".encode()
        if self.kind == "llamacpp":
            final = {"choices": [], "timings": {"prompt_n": n_prompt, "prompt_ms": 50.0, "cache_n": 0,
                                                "predicted_n": n_gen, "predicted_ms": 400.0}}
        else:
            final = {"choices": [], "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen}}
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    # --------------------------------------------------------------- lifecycle

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
