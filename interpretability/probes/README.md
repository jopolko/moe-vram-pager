# Trained probes

Linear probes on the residual stream, one directory per model. The live viewer
(`obench-interp serve`) loads these to produce the Q1..Q5 signals - there are no
regex or keyword classifiers anywhere in the live path.

```
probes/<model_slug>/<name>.npz    w, b, mean, scale  (per layer)
probes/<model_slug>/<name>.json   layer, cv_accuracy, base_rate, classes, ...
```

`<model_slug>` is the HF id with `/` replaced by `__`
(`google__gemma-2-2b-it`). These files are committed so the sidecar works out of
the box.

## Retraining

```
obench-interp train-probes --model google/gemma-2-2b-it
obench-interp train-probes --model Qwen/Qwen2.5-3B-Instruct
```

The model must be in `hf_cache/` already (`obench-interp pull`). Add
`--probe language` / `--probe sycophancy` to train one; `--layer-stride N` to
capture every Nth layer (default 1, ~2 min for a 2B).

## The probes

| name | kind | dataset | what it reads |
|---|---|---|---|
| `language` | multiclass (9 languages) | `data/probe_language.json` | which language the residual stream at a layer decodes to - the "language of thought" when the middle band differs from the prompt |
| `sycophancy` | binary (caved / firm) | `data/probe_sycophancy.json` | whether the state at the end of a pressured prompt precedes the model giving the user the wrong answer they pushed for |

`language` is captured on raw text (no chat template), last token. `sycophancy`
is captured on the chat-templated prompt, last token, and its labels are
**behavioural**: `train-probes` generates an answer for every (question, pressure
style) pair and labels it via `probes.pick_answer` against the known correct /
pushed strings. A well-aligned model that never caves produces one class and the
probe is skipped - check the `train-probes` output.

## Reading the meta JSON

- `layer` - the best cross-validated layer, used for the headline meter.
- `cv_accuracy` - 5-fold CV accuracy at that layer. Below ~0.7 the live panel
  shows the signal as low-confidence.
- `base_rate` - majority-class fraction in the training set (the accuracy floor).
- `layer_accuracy` - CV accuracy per captured layer, for the per-layer strip.
- `dataset_sha` - first 12 hex of the dataset hash the probe was trained on.

## Phase 2

`planning`, `faithfulness`, and `gaming` probes are not built yet; their live
panels are inert and `/experiment/gaming` returns 501. The Q2/Q3 causal tests
(`/experiment/plan`, `/experiment/faithful`) work now.
