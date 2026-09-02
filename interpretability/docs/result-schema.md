# Result schema (all experiments)

Every experiment writes through `report.emit()` (`src/obench_interp/report.py`),
so `results/<experiment>/<timestamp>/results.json` always has this shape:

```json
{
  "schema_version": 1,
  "experiment": "exp1_multilingual",
  "generated_at": "2026-09-01T12:00:00",
  "model": "google/gemma-2-2b",
  "params": { "...": "whatever knobs this run used" },
  "per_item": [ { "...": "one entry per example (concept / couplet / question)" } ],
  "aggregate": { "...": "rolled-up numbers for the whole run" }
}
```

`schema_version` bumps only on a breaking change to these top-level keys.
`per_item` / `aggregate` internals are experiment-specific and documented in
each experiment's own README:

- `experiments/exp1_multilingual/README.md`
- `experiments/exp2_planning/README.md`
- `experiments/exp3_cot_faithfulness/README.md`

A `results/<experiment>/latest` symlink always points at the most recent run
dir. `summary.md` next to `results.json` is the human-readable version of the
same data; a future SvelteKit view is expected to read `results.json` and
render its own summary rather than parse `summary.md`.
