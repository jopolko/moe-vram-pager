# openbench-toolkit / interpretability

A Python mechanistic-interpretability pipeline. **Fully decoupled** from the
llama.cpp / CUDA / GGUF inference path in the rest of this repo: separate
dependency tree, separate venv, no shared build steps, no shared runtime.

## Why separate from the pager code

Interpretability tooling (TransformerLens, SAELens, nnsight) needs raw PyTorch
activations mid-forward-pass. The llama.cpp / GGUF path does not expose that, and
quantization is very likely to distort or destroy the fine-grained activation
patterns this analysis depends on. No existing tooling has validated
SAE / patching techniques against quantized activations. So this module runs the
model via HuggingFace `transformers` in fp16, **not** the quantized GGUF build.

Running this analysis against quantized GGUF weights is an explicit
non-goal / open question. Do not attempt it here without validation work first.

## Research questions

1. Does the model share internal concept representations across languages?
2. Does the model plan multi-token output ahead of time (not just next-token greedy)?
3. Does the model's stated chain-of-thought match what it actually computes
   internally, or does it sometimes fabricate plausible-looking steps?

## Phases

| Phase | Experiment | Status |
|-------|-----------|--------|
| 1 | Multilingual concept sharing (residual-stream feature overlap across languages) | implemented (`run exp1`) |
| 2 | Planning-ahead / activation patching (rhyming couplets) | implemented (`run exp2`) |
| 3 | Chain-of-thought faithfulness via activation patching | implemented (`run exp3`, needs `--instruct` weights) |

Every experiment writes through the same result envelope
(`docs/result-schema.md`) so a future SvelteKit view can read all three
without a migration. No frontend visualization work is done here.

## Hardware

Target dev box: single **GTX 1080 Ti, 11 GB VRAM** (Pascal).

Pascal notes baked into `config.py`:
- No hardware bf16. Use **fp16** (or fp32 on CPU).
- Pascal fp16 *compute* throughput is poor; fp16 here is mainly for memory. For a
  few dozen short prompts this is fine. CPU fallback is viable for 2-3B models.

fp16 memory budget (weights only, before activations / TransformerLens conversion overhead):

| Model | Params | fp16 weights | Fits 11 GB? |
|-------|--------|-------------|-------------|
| google/gemma-2-2b | 2.6 B | ~5.2 GB | yes (conversion peak is tight, CPU-offload the load) |
| Qwen/Qwen2.5-1.5B | 1.5 B | ~3.1 GB | yes, comfortably |
| Qwen/Qwen2.5-3B | 3.1 B | ~6.2 GB | yes |
| Qwen/Qwen2.5-7B | 7.6 B | ~15 GB | no (needs 8-bit or a bigger card) |

Disk: model weights 3-6 GB, plus pretrained SAEs (a few GB if applicable).

## Model choice

Confirmed: `google/gemma-2-2b` (+ `-it` for phase 3). See `docs/model-choice.md`
for the reasoning.

## Setup

```bash
cd interpretability
uv venv .venv
uv pip install -e ".[dev]"
```

### GPU note (Pascal / GTX 10-series dev box)

`uv pip install -e .` pulls the newest CPU-or-CUDA `torch` its resolver picks,
which on a fresh install is a CUDA build too new for the GTX 1080 Ti - PyTorch
dropped Pascal (`sm_61`) offline kernels in the 2.8 `cu128`/`cu129` wheels. The
last wheel that still ships Pascal kernels **and** satisfies this project's
`transformer_lens>=2.6` floor is **torch 2.7.x on the `cu126` index**:

```bash
uv pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
```

Nothing else in the dependency tree needs to move - the interp libs floor at
`torch>=2.4-2.6`, none ceiling it. On a Turing-or-newer GPU (RTX 20xx →
50-series) skip this and let the default resolve stand.

## Usage

Wrappers activate the venv, pin the HF cache into this directory, and source the
HF token (checked in order: `/var/secrets/nowservingto.env`,
`~/secrets/openbench.env`, `~/secrets/nowservingto.env`):

- **Linux / Git Bash:** `./interp <cmd>`
- **PowerShell:** `.\interp.ps1 <cmd>`

```bash
./interp doctor              # env / GPU / weight readiness
./interp pull                # gemma-2-2b + Gemma Scope SAE (layer 12, ~6 GB)
./interp pull --instruct     # also gemma-2-2b-it, for phase 3
./interp list                # phases and their status
./interp run exp1            # experiment 1: multilingual concept sharing
./interp run exp1 --layer 10 --top-k 48 --languages en fr de es
./interp run exp2            # experiment 2: planning ahead (rhyming couplets)
./interp run exp2 --model Qwen/Qwen2.5-3B-Instruct   # any HF decoder
./interp run exp3            # experiment 3: CoT faithfulness
./interp run exp3 --model Qwen/Qwen2.5-3B-Instruct --layer 18
```

Results land in `results/<experiment>/<timestamp>/` as `results.json` (stable
schema, see `docs/result-schema.md`) + `summary.md`, with a `latest` symlink.
Each experiment's method and how to read its output: `experiments/exp1_multilingual/README.md`,
`experiments/exp2_planning/README.md`, `experiments/exp3_cot_faithfulness/README.md`.

### Backends

- **exp1** uses **TransformerLens** — it is bound to the Gemma Scope SAEs, so it
  only runs on `google/gemma-2-2b`.
- **exp2 / exp3** use **nnsight** (`src/obench_interp/activations.py`) and take a
  `--model <hf-id>` flag. nnsight wraps the HF model as-is (no TransformerLens
  weight-conversion, which is a ~2x transient RAM peak that OOMs a 24 GB box on
  a 2B model), and works on any `model.model.layers[i]` decoder — gemma-2,
  Qwen2.5/3, Llama-3, Mistral. `--layer` defaults to the model's middle layer.
  These need an **instruct** model that actually writes couplets / chains of
  thought; a 0.5–2B base model produces neither. exp2's SAE feature step only
  runs on `google/gemma-2-2b`.

Confirmed model (2026-09-01): **`google/gemma-2-2b`**, layer 12 residual stream,
Gemma Scope `gemma-scope-2b-pt-res-canonical` width-16k SAE. See
`docs/model-choice.md` for the reasoning.

## Viewer (webui `#/interp`)

The chat webui (`tools/ui/`) has an **Interpretability** tab (flask icon) that
renders these runs — cosine heatmaps for exp1, per-couplet
baseline/patched/control panels for exp2, side-by-side chains of thought with
the faithfulness verdict for exp3.

Open the tab and click **Open results folder**, then pick this
`interpretability/results/` directory. That's it — the browser reads the
`results.json` files directly (File System Access API, Chrome/Edge), and the
folder is remembered for next time. New `obench-interp run`s show up on Rescan.

No folder picker (Firefox), or want it headless? Two fallbacks in the same tab:

- **drop** `results.json` files onto the page, or
- **"connect to a running sidecar"** — `python tools/interp_ui_api.py` starts a
  stdlib-only (no `pip install`), read-only, loopback server on `:8087` that
  serves this directory; same pattern as the pentest tab's `pentest_ui_api.py`.
