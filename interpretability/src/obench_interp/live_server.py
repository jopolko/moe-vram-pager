"""Loopback HTTP + SSE sidecar for the webui's live interpretability chat.

Same role as `tools/interp_ui_api.py` (which serves the batch `run exp*` result
envelopes), but this one holds a loaded fp16 model and streams a token at a
time. It runs inside this package's venv -- it imports nnsight through
`live.LiveSession` -- so it ships as `obench-interp serve` and the `interp` /
`interp.ps1` wrappers pick up the venv and HF token for it.

Every live signal is a trained probe (see probes.py); panels light up only for
models with probes (`obench-interp train-probes --model <id>`).

Stdlib only, loopback only, no auth (same posture as the rest of the appliance).

    obench-interp serve                 # 127.0.0.1:8088
    obench-interp serve --port 9000

Routes:
    GET  /health              loaded model + layer + probe count, or nothing loaded
    GET  /models              causal-LM repos in the HF cache, tagged (incl. `probes`)
    GET  /probes              probe metadata for the loaded model
    GET  /feature/<id>        SAE feature: local top words + Neuronpedia description
    POST /load   {model,layer?,device?}   load a model (blocks); device 'cpu' | 'cuda'
    POST /chat   {messages,max_new_tokens?,capture_stride?,layer?,hint?}   SSE token stream
    POST /experiment/decide     {question,correct,wrong,max_new_tokens?}           forking-paths decision trace
    POST /experiment/plan       {prompt_a,prompt_b,max_new_tokens?}                exp2 causal patch
    POST /experiment/faithful   {question,hint,hint_answer?,correct?,...}          exp3 causal ablation
    POST /experiment/sycophancy {question,pressure,correct_answer?,pushed_answer?} pressure-frame ablation
    POST /experiment/gaming     501 until phase 2
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .env import HF_CACHE
from .live import DEFAULT_MAX_NEW_TOKENS, SAE_MODELS, list_cached_models

# The live viewer only ever runs models already in the HF cache (`interp pull`
# adds them). Skip the per-file etag HEAD requests to huggingface.co that
# `from_pretrained` makes on every load -- on a slow link they block ~30 s with
# nothing touching disk or VRAM. Set before transformers is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"])
    return str(messages[-1].get("content", "")) if messages else ""


# One model in memory at a time; the lock also serializes generation so two
# chat requests can't interleave trace calls on the same model.
_session = None
_lock = threading.Lock()


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _ensure_session(model: str | None, layer: int | None, device: str | None = None):
    """Return a `LiveSession` for `model`, loading/reloading if needed."""
    global _session
    from .live import LiveSession

    reload = model and (
        _session is None
        or _session.model_id != model
        or (device is not None and _session.device != device)
    )
    if reload:
        _session = None
        _free_gpu()
        _session = LiveSession(model, layer=layer, device=device)
    elif _session is None:
        raise RuntimeError("no model loaded; POST /load first")
    else:
        _session.set_layer(layer)
    return _session


class Handler(BaseHTTPRequestHandler):
    server_version = "interp-live-api/2.0"
    protocol_version = "HTTP/1.1"

    # ---- response helpers ----
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _sse_open(self) -> None:
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()

    def _sse(self, obj: dict) -> None:
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt: str, *args) -> None:  # quieter
        return

    # ---- routing ----
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("", "/health"):
            s = _session
            return self._json(
                200,
                {
                    "ok": True,
                    "cache": str(HF_CACHE),
                    "model": getattr(s, "model_id", None),
                    "layer": getattr(s, "layer", None),
                    "n_layers": getattr(s, "n_layers", None),
                    "device": getattr(s, "device", None),
                    "probes": [p["name"] for p in s.probe_meta()] if s else [],
                },
            )
        if path == "/models":
            return self._json(200, {"models": list_cached_models()})
        if path == "/probes":
            s = _session
            if s is None:
                return self._json(409, {"error": "no model loaded"})
            return self._json(200, {"model": s.model_id, "probes": s.probe_meta()})
        if path == "/trust-bank":
            try:
                from .env import INTERP_ROOT

                data = json.loads(
                    (INTERP_ROOT / "data" / "trust_questions.json").read_text(encoding="utf-8")
                )
                return self._json(200, {"items": data.get("items", [])})
            except (OSError, ValueError) as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        parts = path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "feature":
            s = _session
            if s is None:
                return self._json(409, {"error": "no model loaded"})
            try:
                fid = int(parts[1])
            except ValueError:
                return self._json(400, {"error": "feature id must be an integer"})
            return self._json(200, s.feature_description(fid))
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/load":
            return self._handle_load()
        if path == "/chat":
            return self._handle_chat()
        if path == "/experiment/plan":
            return self._handle_experiment("plan")
        if path == "/experiment/faithful":
            return self._handle_experiment("faithful")
        if path == "/experiment/gaming":
            return self._json(501, {"error": "the Q4 gaming probe + causal test arrive in phase 2"})
        if path == "/experiment/sycophancy":
            return self._handle_experiment("sycophancy")
        if path == "/experiment/decide":
            return self._handle_experiment("decide")
        if path == "/experiment/trust":
            return self._handle_experiment("trust")
        self._json(404, {"error": "not found"})

    def _handle_load(self) -> None:
        body = self._read_json()
        model = body.get("model")
        if not model:
            return self._json(400, {"error": "missing 'model'"})
        device = (body.get("device") or "").strip().lower() or None
        if device not in (None, "cpu", "cuda"):
            return self._json(400, {"error": "device must be 'cpu' or 'cuda'"})
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                return self._json(400, {"error": "device 'cuda' requested but no CUDA device"})
        try:
            with _lock:
                s = _ensure_session(model, body.get("layer"), device)
        except Exception as e:  # loading surfaces plenty of failure modes
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        self._json(
            200,
            {
                "ok": True,
                "model": s.model_id,
                "layer": s.layer,
                "n_layers": s.n_layers,
                "device": s.device,
                "probes": s.probe_meta(),
            },
        )

    def _handle_chat(self) -> None:
        body = self._read_json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._json(400, {"error": "missing 'messages'"})
        max_new = body.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS
        hint = (body.get("hint") or "").strip()
        prompt_text = _last_user_text(messages)

        acquired = _lock.acquire(blocking=False)
        if not acquired:
            return self._json(409, {"error": "busy: another generation is running"})
        try:
            try:
                dev = (body.get("device") or "").strip().lower() or None
                s = _ensure_session(body.get("model"), body.get("layer"), dev)
            except RuntimeError as e:
                return self._json(409, {"error": str(e)})
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

            # a biasing "hint" (planted fact) is prepended to the last user turn;
            # it primes the /experiment/faithful test and the user sees the reply move
            send_messages = [dict(m) for m in messages]
            if hint:
                for m in reversed(send_messages):
                    if m.get("role") == "user":
                        m["content"] = f"{hint} {m.get('content', '')}".strip()
                        break

            input_ids = s.render_prompt(send_messages)
            stride = max(1, int(body.get("capture_stride") or body.get("lens_stride") or 1))
            pl = s.prompt_language(prompt_text)  # caches the baseline used by q1

            self._sse_open()
            self._sse(
                {
                    "type": "start",
                    "model": s.model_id,
                    "layer": s.layer,
                    "n_layers": s.n_layers,
                    "prompt_tokens": int(input_ids.shape[1]),
                    "prompt_lang": (pl or {}).get("surface_lang"),
                    "probes": s.probe_meta(),
                    "sae": s.model_id in SAE_MODELS,
                    "hint": bool(hint),
                }
            )

            t0 = time.monotonic()
            text_parts: list[str] = []
            n = 0
            try:
                for step in s.generate(input_ids, max_new, capture_stride=stride):
                    n = step.index + 1
                    text_parts.append(step.text)
                    self._sse(
                        {
                            "type": "token",
                            "index": step.index,
                            "token": step.text,
                            "token_id": step.token_id,
                            "finished": step.finished,
                            "q1": step.q1,
                            "q2": step.q2,
                            "q3": step.q3,
                            "q4": step.q4,
                            "q5": step.q5,
                            "surprisal": step.surprisal,
                            "sae": step.sae,
                        }
                    )
            except (BrokenPipeError, ConnectionResetError):
                return  # client hung up; stop generating
            except Exception as e:
                self._sse({"type": "error", "detail": f"{type(e).__name__}: {e}"})
                return

            elapsed = time.monotonic() - t0
            self._sse(
                {
                    "type": "done",
                    "text": "".join(text_parts),
                    "tokens": n,
                    "elapsed_s": round(elapsed, 2),
                    "tok_per_s": round(n / elapsed, 2) if elapsed > 0 else None,
                }
            )
        finally:
            _lock.release()

    def _handle_experiment(self, kind: str) -> None:
        body = self._read_json()
        max_new = body.get("max_new_tokens")
        if kind == "plan":
            a, b = (body.get("prompt_a") or "").strip(), (body.get("prompt_b") or "").strip()
            if not a or not b:
                return self._json(400, {"error": "need 'prompt_a' and 'prompt_b'"})
        elif kind == "sycophancy":
            q, pressure = (body.get("question") or "").strip(), (body.get("pressure") or "").strip()
            if not q or not pressure:
                return self._json(400, {"error": "need 'question' and 'pressure'"})
        elif kind in ("decide", "trust"):
            q = (body.get("question") or "").strip()
            correct = (body.get("correct") or "").strip()
            wrong = (body.get("wrong") or body.get("other") or "").strip()
            if not q or not correct or not wrong:
                return self._json(400, {"error": "need 'question', 'correct' and 'wrong'/'other'"})
        else:
            q, h = (body.get("question") or "").strip(), (body.get("hint") or "").strip()
            if not q or not h:
                return self._json(400, {"error": "need 'question' and 'hint'"})

        with _lock:
            s = _session
            if s is None:
                return self._json(409, {"error": "no model loaded; POST /load first"})
            try:
                if kind == "plan":
                    out = s.planning_test(a, b, max_new_tokens=max_new or 24)
                elif kind == "decide":
                    out = s.decision_trace(q, correct, wrong, max_new_tokens=max_new or 140)
                elif kind == "trust":
                    out = s.trust_answer(
                        q, correct, wrong,
                        n_samples=int(body.get("n_samples") or 24),
                        max_new_tokens=max_new or 140,
                    )
                elif kind == "sycophancy":
                    out = s.sycophancy_test(
                        q,
                        pressure,
                        correct_answer=(body.get("correct_answer") or "").strip() or None,
                        pushed_answer=(body.get("pushed_answer") or "").strip() or None,
                        max_new_tokens=max_new or 160,
                    )
                else:
                    out = s.faithfulness_test(
                        q,
                        h,
                        max_new_tokens=max_new or 96,
                        hint_answer=(body.get("hint_answer") or "").strip() or None,
                        correct=(body.get("correct") or "").strip() or None,
                    )
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        self._json(200, {"ok": True, "experiment": kind, "model": s.model_id, "result": out})


def _prewarm() -> None:
    """Import torch / nnsight off the request path so the first /load only pays
    for weights, not the one-time cold import."""
    try:
        import torch

        torch.cuda.is_available()
        from nnsight import LanguageModel  # noqa: F401

        print("  prewarm: torch + nnsight ready")
    except Exception as e:
        print(f"  prewarm skipped: {type(e).__name__}: {e}")


def serve(host: str = "127.0.0.1", port: int = 8088) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    n = len(list_cached_models())
    print(f"interp-live-api  ->  http://{host}:{port}")
    print(f"  HF cache {HF_CACHE}  ({n} cached causal-LM repo{'s' if n != 1 else ''})")
    print("  open the webui #/interp and switch to Live")
    threading.Thread(target=_prewarm, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="obench-interp serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    args = ap.parse_args(argv)
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
