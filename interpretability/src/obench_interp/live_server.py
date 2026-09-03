"""Loopback HTTP + SSE sidecar for the webui's live interpretability chat.

Same role as `tools/interp_ui_api.py` (which serves the batch `run exp*`
result envelopes), but this one holds a loaded fp16 model and streams a token
at a time. It has to run inside this package's venv -- it imports nnsight
through `live.LiveSession` -- so it ships as `obench-interp serve` and the
`interp` / `interp.ps1` wrappers pick up the venv and HF token for it.

Stdlib only, loopback only, no auth (same posture as the rest of the appliance).

    obench-interp serve                 # 127.0.0.1:8088
    obench-interp serve --port 9000

Routes:
    GET  /health              loaded model + layer, or nothing loaded
    GET  /models              causal-LM repos in the HF cache, tagged
    GET  /feature/<id>        SAE feature: local top words + Neuronpedia description
    POST /load   {model,layer?,device?}   load a model (blocks); device 'cpu' | 'cuda'
    POST /chat   {messages,max_new_tokens?,layer?,model?,hint?,visible_tests?}   SSE token stream
    POST /experiment/plan     {prompt_a,prompt_b,max_new_tokens?}            exp2 causal patch
    POST /experiment/faithful {question,hint,hint_answer?,correct?,...}      exp3 causal ablation
    POST /experiment/gaming   {task,pressure,visible_tests?,max_new_tokens?} exp4 causal ablation
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .env import HF_CACHE
from .live import (
    DEFAULT_MAX_NEW_TOKENS,
    SAE_MODELS,
    detect_language,
    faithfulness_watch,
    gaming_watch,
    language_readout,
    list_cached_models,
    planning_readout,
)


def _int_list(v) -> list[int]:
    """Coerce a `visible_tests` field to a flat list of int inputs.

    Accepts "2, 3, 17", [2, 3, 17], or [[2, True], [3, True]] (first of each pair).
    """
    if isinstance(v, str):
        v = re.findall(r"-?\d+", v)
    if not isinstance(v, list):
        return []
    out: list[int] = []
    for x in v:
        if isinstance(x, (list, tuple)) and x:
            x = x[0]
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            pass
    return out


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"])
    return str(messages[-1].get("content", "")) if messages else ""


# One model in memory at a time; the lock also serializes generation so two
# chat requests can't interleave trace calls on the same model.
_session = None
_lock = threading.Lock()


def _ensure_session(model: str | None, layer: int | None, device: str | None = None):
    """Return a `LiveSession` for `model`, loading/reloading if needed.

    A different `model` or a different `device` than the live session forces a
    reload; `device=None` keeps whatever the session already has.
    """
    global _session
    from .live import LiveSession

    reload = model and (
        _session is None
        or _session.model_id != model
        or (device is not None and _session.device != device)
    )
    if reload:
        _session = None  # free the old one before allocating the new
        _session = LiveSession(model, layer=layer, device=device)
    elif _session is None:
        raise RuntimeError("no model loaded; POST /load first")
    else:
        _session.set_layer(layer)
    return _session


class Handler(BaseHTTPRequestHandler):
    server_version = "interp-live-api/1.0"
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
        # No Content-Length on a stream, so the socket must close to end it;
        # the webui's EventSource treats that as a normal end-of-response.
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
                },
            )
        if path == "/models":
            return self._json(200, {"models": list_cached_models()})
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
            return self._handle_experiment("gaming")
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
            },
        )

    def _handle_chat(self) -> None:
        body = self._read_json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._json(400, {"error": "missing 'messages'"})
        max_new = body.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS
        hint = (body.get("hint") or "").strip()
        visible = _int_list(body.get("visible_tests"))

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

            # A biasing "hint" is prepended to the last user turn for generation;
            # the Q3 watch then tracks whether the reply ever acknowledges it.
            send_messages = [dict(m) for m in messages]
            if hint:
                for m in reversed(send_messages):
                    if m.get("role") == "user":
                        m["content"] = f"{hint} {m.get('content', '')}".strip()
                        break

            input_ids = s.render_prompt(send_messages)
            prompt_text = _last_user_text(messages)
            stride = max(1, int(body.get("lens_stride") or 1))
            self._sse_open()
            self._sse(
                {
                    "type": "start",
                    "model": s.model_id,
                    "layer": s.layer,
                    "n_layers": s.n_layers,
                    "prompt_tokens": int(input_ids.shape[1]),
                    "prompt_lang": detect_language(prompt_text),
                    "sae": s.model_id in SAE_MODELS,
                    "hint": bool(hint),
                    "gaming": bool(visible),
                }
            )

            t0 = time.monotonic()
            text_parts: list[str] = []
            history: list[dict] = []
            n = 0
            try:
                for step in s.generate(input_ids, max_new, lens_stride=stride):
                    n = step.index + 1
                    text_parts.append(step.text)
                    output_text = "".join(text_parts)
                    history.append(
                        {
                            "index": step.index,
                            "token_id": step.token_id,
                            "text": step.text,
                            "next_top": step.next_top,
                        }
                    )
                    self._sse(
                        {
                            "type": "token",
                            "index": step.index,
                            "token": step.text,
                            "token_id": step.token_id,
                            "finished": step.finished,
                            "q1": language_readout(step.lens, prompt_text, output_text, s.n_layers),
                            "q2": planning_readout(history),
                            "q3": faithfulness_watch(output_text, hint) if hint else None,
                            "q4": gaming_watch(output_text, visible) if visible else None,
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
        elif kind == "gaming":
            task, pressure = (body.get("task") or "").strip(), (body.get("pressure") or "").strip()
            if not task or not pressure:
                return self._json(400, {"error": "need 'task' and 'pressure'"})
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
                elif kind == "gaming":
                    out = s.gaming_test(
                        task,
                        pressure,
                        visible_values=_int_list(body.get("visible_tests")),
                        max_new_tokens=max_new or 200,
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


def serve(host: str = "127.0.0.1", port: int = 8088) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    n = len(list_cached_models())
    print(f"interp-live-api  ->  http://{host}:{port}")
    print(f"  HF cache {HF_CACHE}  ({n} cached causal-LM repo{'s' if n != 1 else ''})")
    print("  open the webui #/interp and switch to Live")
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
