#!/usr/bin/env python3
"""Read-only HTTP sidecar for the webui's #/interp viewer.

Serves `interpretability/results/<experiment>/<timestamp>/results.json` -- the
envelopes the `obench-interp run` CLI writes -- to the static SPA, which fetches
them cross-origin the same way #/pentest talks to pentest_ui_api.py.

Deliberately stdlib-only (no fastapi/uvicorn): it just reads JSON files, so it
should run with any Python 3 and no `pip install`. Loopback-only, no auth.

    python tools/interp_ui_api.py            # port 8087
    python tools/interp_ui_api.py --port 9000 --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_DIR / "interpretability" / "results"
EXPERIMENTS = ("exp1_multilingual", "exp2_planning", "exp3_cot_faithfulness")
_TS_RE = re.compile(r"^[0-9]{8}-[0-9]{6}$")


def _headline(experiment: str, agg: dict) -> str:
    try:
        if experiment == "exp1_multilingual":
            return (
                f"cosine {agg.get('mean_cross_language_cosine', '?')} · "
                f"{agg.get('language_agnostic_feature_count', 0)} shared features"
            )
        if experiment == "exp2_planning":
            return (
                f"planning effect {agg.get('planning_effect', '?')} "
                f"({agg.get('planning_flip_rate', '?')} vs {agg.get('control_flip_rate', '?')} control)"
            )
        if experiment == "exp3_cot_faithfulness":
            return (
                f"{agg.get('unfaithful_count', 0)}/{agg.get('n_items', 0)} unfaithful · "
                f"follow {agg.get('hint_follow_rate', '?')}"
            )
    except Exception:
        pass
    return ""


def _run_files() -> list[tuple[str, str, Path]]:
    if not RESULTS_DIR.is_dir():
        return []
    out: list[tuple[str, str, Path]] = []
    for experiment in EXPERIMENTS:
        exp_dir = RESULTS_DIR / experiment
        if not exp_dir.is_dir():
            continue
        for run in sorted((p for p in exp_dir.iterdir() if p.is_dir()), reverse=True):
            results = run / "results.json"
            if results.is_file():
                out.append((experiment, run.name, results))
    return out


def _list_runs() -> dict:
    runs = []
    for experiment, timestamp, results in _run_files():
        try:
            data = json.loads(results.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        agg = data.get("aggregate", {}) or {}
        runs.append(
            {
                "experiment": experiment,
                "timestamp": timestamp,
                "generated_at": data.get("generated_at"),
                "model": data.get("model"),
                "layer": (data.get("params") or {}).get("layer"),
                "backend": (data.get("params") or {}).get("backend", "transformer_lens"),
                "n_items": agg.get("n_items") or len(data.get("per_item") or []),
                "headline": _headline(experiment, agg),
                "aggregate": agg,
            }
        )
    return {"results_dir": str(RESULTS_DIR), "runs": runs}


def _all_runs() -> dict:
    """Every run's full results.json, each tagged with its `_timestamp` dir name.

    One call, so the webui's folder / drop / sidecar paths all load the same way.
    """
    out = []
    for experiment, timestamp, results in _run_files():
        try:
            data = json.loads(results.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        data["_timestamp"] = timestamp
        out.append(data)
    return {"results_dir": str(RESULTS_DIR), "runs": out}


def _get_run(experiment: str, timestamp: str) -> dict | None:
    if experiment not in EXPERIMENTS or not _TS_RE.match(timestamp):
        return None
    results = RESULTS_DIR / experiment / timestamp / "results.json"
    if not results.is_file():
        return None
    try:
        return json.loads(results.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "interp-ui-api/1.0"

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("", "/health"):
            files = _run_files()
            return self._send(200, {"ok": True, "results_dir": str(RESULTS_DIR), "runs": len(files)})
        if path == "/runs":
            return self._send(200, _list_runs())
        if path == "/all":
            return self._send(200, _all_runs())
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "runs":
            run = _get_run(parts[1], parts[2])
            if run is None:
                return self._send(404, {"error": "run not found"})
            return self._send(200, run)
        self._send(404, {"error": "not found"})

    def log_message(self, fmt: str, *args) -> None:  # quieter
        return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8087)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    n = len(_run_files())
    print(f"interp-ui-api  ->  http://{args.host}:{args.port}")
    print(f"  serving {RESULTS_DIR}  ({n} run{'s' if n != 1 else ''})")
    print("  open the webui and go to #/interp")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
