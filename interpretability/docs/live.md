# Live viewer

`obench-interp serve` (see the README) starts a loopback sidecar that loads one
fp16 HF model and streams a token at a time to the webui's `#/interp` -> **Live**
tab. Unlike the batch `run exp*` path, the prompt is whatever you type.

Every live signal is a **trained linear probe** on the residual stream. There
are no regex or keyword classifiers over the generated text. A probe is fit
offline by `obench-interp train-probes` on activations captured from a labelled
prompt set (`data/probe_*.json`); the sidecar loads the set for the model it has
open and scores each generated token. Panels light up only for models that have
that probe.

Phase 1 ships **Q1** (language of thought) and **Q5** (caving to pressure). Q2 /
Q3 / Q4 probes come next; their causal tests already run.

## Training the probes

```
obench-interp train-probes --model google/gemma-2-2b-it
obench-interp train-probes --model Qwen/Qwen2.5-3B-Instruct
```

Writes `probes/<model_slug>/<name>.{npz,json}` (committed). `--probe language`
or `--probe sycophancy` trains just one; `--layer-stride N` captures every Nth
layer (default 1). The model must already be in `hf_cache/` (`obench-interp
pull`). See `probes/README.md`.

## Endpoints

| method | path | body | returns |
|---|---|---|---|
| GET | `/health` | - | loaded model + probe layer + probe names, or nulls |
| GET | `/models` | - | causal-LM repos in `hf_cache/`, each `{name,n_layers,instruct,sae,probes}` |
| GET | `/probes` | - | probe metadata for the loaded model |
| GET | `/feature/<id>` | - | one SAE feature: locally-decoded top `words` + a Neuronpedia `description` |
| POST | `/load` | `{model, layer?, device?}` | loads it (blocks); `{model,layer,n_layers,device,probes}` |
| POST | `/chat` | `{messages, max_new_tokens?, capture_stride?, layer?, hint?}` | SSE stream |
| POST | `/experiment/plan` | `{prompt_a, prompt_b, max_new_tokens?}` | exp2 causal patch |
| POST | `/experiment/faithful` | `{question, hint, hint_answer?, correct?, max_new_tokens?}` | exp3 causal ablation |
| POST | `/experiment/sycophancy` | `{question, pressure, correct_answer?, pushed_answer?, max_new_tokens?}` | pressure-frame ablation |
| POST | `/experiment/gaming` | - | 501 until phase 2 |

`/chat` SSE events: `start`, then one `token` per generated token, then `done`
(or `error`). The `start` event carries `probes` (metadata: name, layer,
cv_accuracy, base_rate, classes) and `prompt_lang` (the language probe's read on
the raw prompt). Each `token` event carries `q1`..`q5` (the probe readouts, or
null where no probe is trained), `surprisal` (bits for the chosen token) and, on
`gemma-2-2b(-it)`, `sae`.

## Q1: what language is it "thinking" in?

The **language probe** is a multiclass logistic regression on the standardized
residual stream, one per layer, trained on `data/probe_language.json` (short
everyday sentences in en/fr/de/es/it/pt/ru/zh/ja). For every generated token the
sidecar scores each captured layer:

- `internal_lang` = the plurality probe prediction over the middle band of
  layers (`probes.LANG_BAND`, 35-80% of depth), with `internal_confidence` = the
  fraction of band layers that agreed;
- `surface_lang` = the probe prediction at its best CV layer;
- `prompt_lang` = the probe's read on the raw user text (no chat template);
- `shared_concept_space` = `internal_lang` differs from `prompt_lang`: the
  concept is represented in a language the surface text is not in.

`layers` is the per-layer strip: `{layer, lang, p}` for each captured layer.
`cv_accuracy` (from the `start` event's `probes`) is the probe's held-out
accuracy on the training set for this model - a low number means the probe is
not reliable here and the panel says so.

The causal evidence for shared multilingual representations is `run exp1` (SAE
feature overlap across languages); the live probe is a per-prompt read of the
same question.

### SAE features (gemma-2-2b / -it)

Unchanged: when the loaded model has a Gemma Scope SAE for the probe layer, each
`token` event lists the top firing SAE features, tagging any that a prior
`run exp1` marked language-agnostic. `GET /feature/<id>` adds a Neuronpedia
description + `max_act`. See the panel copy for details.

## Q2: planning ahead

Phase 2. The planning probe (future-token decodability: how many tokens ahead a
line's ending is already linearly readable) is not trained yet, so the live
panel is inert.

**Causal test** (`POST /experiment/plan {prompt_a, prompt_b}`): exp2's patch on
one pair. Both prompts are tokenised as raw completions; generate from A, then
splice B's residual stream in at the last-prompt-token ("planning") position
and, as a control, at an early position. If only the planning splice changes A's
ending, A had committed to that ending there. Small / base models often show no
effect - `gemma-2-2b` is the reference.

## Where it decided (forking-paths decision trace)

`POST /experiment/decide {question, correct, wrong, max_new_tokens?}` -
`LiveSession.decision_trace`, an implementation of Bigelow et al.'s "forking
paths" on one item. No probe, no heuristic - pure generation.

1. Generate the chain of thought greedily, recording the top-2 tokens at each
   step.
2. At a strided set of positions (`stride` 3, capped at `max_points` 22), force
   the 2nd-choice token and regenerate the tail; classify each run with
   `probes.pick_answer` against `correct` / `wrong`.
3. `commit_index` = the earliest position after which no single-token swap
   changes the final answer - the reasoning's point of no return. Each `points`
   entry says whether forcing the runner-up there flips the answer and, if so,
   to which of the two. `pivot` is the last still-pivotal position.

If the greedy answer is `wrong` and a pivot exists whose `alt_answer` is
`correct`, that token is exactly where it went wrong and the right path was one
word away. If nothing flips anywhere and the answer is right, the model had it
locked from the first token. `points` all grey with a wrong answer means the
correct answer was never among the top alternatives.

~20 forked continuations, ~1-3 min on a 2B. Returns JSON, no run dir.

## Q3: CoT faithfulness

Phase 2. The "answer known before reasoning" probe is not trained yet.

**Causal test** (`POST /experiment/faithful {question, hint, hint_answer,
correct}`): exp3's ablation on one item. Answer the question alone, with the
`hint` (a planted misleading fact) prepended, and with the hint span's
activations replaced by filler. The verdict needs both `hint_answer` (what the
hint pushes) and `correct`; the comparison is `probes.pick_answer` equality
(whole-word containment against those two known strings, no regex). `hint_driven`
= the hint changed the answer and ablating its activations restored the unhinted
answer: the hint drove the answer at the activation level.

Every causal endpoint runs 3-4 full generations back to back - slow, holds the
model lock, meant for a button not a loop. They return JSON to the Live panel;
they do not write a `results/` run dir.

## Q4: specification gaming

Phase 2. The spec-gaming probe (labelled by exp4's held-out-test oracle) is not
trained yet, and `/experiment/gaming` returns 501. Use `run exp4` for the batch
causal analysis in the meantime.

## Q5: sycophancy (caving to pressure)

The **sycophancy probe** is a binary logistic regression on the residual stream
(caved vs firm), trained on `data/probe_sycophancy.json`: each factual question
is prompted with a pressure preamble pushing a wrong answer, the model generates,
and the item is labelled by which answer it landed on (`probes.pick_answer`
against the known correct / pushed strings). The probe reads the last
prompt-token residual - the state that precedes the first answer token. A model
that rarely caves yields one class and the probe is skipped for it (the panel
says so).

Each `token` event's `q5` carries `p_cave` (probability of caving at the best
layer), `p_cave_prompt` (the same read on the prompt before any answer),
`leaning` (`p_cave >= probes.CAVE_THRESHOLD`), `cv_accuracy`, `base_rate`, and
`layers` (the per-layer `p_cave` trace).

**Causal test** (`POST /experiment/sycophancy {question, pressure,
correct_answer?, pushed_answer?}`): the pressure-frame ablation, same shape as
the Q3 test. Answer `question` three ways - alone, with `pressure` prepended, and
with the pressure span's activations replaced by filler. With `correct_answer` +
`pushed_answer` the verdict is `pick_answer` equality across the three runs:
`sycophantic` = the pressure changed the answer to the pushed one and ablating
its activations flipped it back. The sycophancy probe also reports its read on
the plain vs pressured prompt state (`result.probe`) as a mechanistic
cross-check. A well-aligned model usually holds firm - that is a result, not a
failure. There is no batch `run exp5`; this is Live-only.

## Cost

The streaming loop runs the HF model directly (not `nnsight.trace`) with a KV
cache: the prompt is one full forward pass, each step after it is a single
token. Forward hooks capture the probed layers' last-position residuals; each
probe is a matrix-vector product per captured layer. With the logit lens gone
this is cheap - generation is bounded by tok/s (~9-10 for a 2B on a 1080 Ti).
`capture_stride` thins the per-layer strip if you want it faster.
`torch.cuda.empty_cache()` runs after each turn so `llama-server` gets the VRAM
back. Still an inspection tool, not a throughput one.

## Device

`/load` takes `device: "cpu" | "cuda"` (default: auto = CUDA if present). CPU
keeps the whole interp model off the GPU so `llama-server` can keep its VRAM,
but it loads in fp32 (~10 GB RAM for a 2B model) and generation runs well under
1 tok/s. Switching device on an already-loaded model triggers a reload. The Live
tab has a GPU / CPU toggle next to the model picker.
