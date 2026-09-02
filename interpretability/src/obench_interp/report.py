"""Result emission: structured JSON + a human-readable markdown summary.

Every experiment writes through `emit()` so all three end up with the same
top-level envelope. See `interpretability/docs/result-schema.md` for the schema
this exists to keep stable for a future SvelteKit view.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .env import INTERP_ROOT

RESULTS_DIR = INTERP_ROOT / "results"

SCHEMA_VERSION = 1


def new_run_dir(experiment: str) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = RESULTS_DIR / experiment / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def write(run_dir: Path, results: dict, summary_md: str) -> None:
    """Low-level primitive: write results.json + summary.md, refresh `latest`."""
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    latest = run_dir.parent / "latest"
    latest.unlink(missing_ok=True)
    try:
        latest.symlink_to(run_dir.name)
    except OSError:
        pass


def envelope(experiment: str, *, model: str, params: dict, per_item: list, aggregate: dict) -> dict:
    """Build the shared results dict (not written yet).

    Fixed top-level keys across all experiments:
      schema_version, experiment, generated_at, model, params, per_item, aggregate
    `params` holds whatever knobs the experiment was run with (layer, SAE id,
    languages, hint styles, ...). `per_item` is the list of per-example results
    (per-concept, per-couplet, per-question, ...). `aggregate` is the rolled-up
    summary numbers.

    Callers build a `summary_md` from the returned dict, then call `write()`.
    Split out from writing because most `summary_md` builders want to read the
    finished envelope (e.g. `r["aggregate"][...]`) rather than re-derive it.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "params": params,
        "per_item": per_item,
        "aggregate": aggregate,
    }


def emit(
    experiment: str,
    run_dir: Path,
    *,
    model: str,
    params: dict,
    per_item: list,
    aggregate: dict,
    summary_md,
) -> dict:
    """Build the envelope, write it, and return the dict written.

    `summary_md` is either a markdown string, or a callable `(results) -> str`
    for the common case where the summary needs the finished envelope.
    """
    results = envelope(experiment, model=model, params=params, per_item=per_item, aggregate=aggregate)
    md = summary_md(results) if callable(summary_md) else summary_md
    write(run_dir, results, md)
    return results
