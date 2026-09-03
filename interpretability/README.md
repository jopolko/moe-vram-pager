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
| 4 | Specification gaming / reward hacking under test-passing pressure | implemented (`run exp4`, wants a code-capable instruct model) |

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
./interp run exp4            # experiment 4: specification gaming (default Qwen2.5-3B-Instruct)
```

Results land in `results/<experiment>/<timestamp>/` as `results.json` (stable
schema, see `docs/result-schema.md`) + `summary.md`, with a `latest` symlink.
Each experiment's method and how to read its output: `experiments/exp1_multilingual/README.md`,
`experiments/exp2_planning/README.md`, `experiments/exp3_cot_faithfulness/README.md`,
`experiments/exp4_gaming/README.md`.

### Backends

- **exp1** uses **TransformerLens** — it is bound to the Gemma Scope SAEs, so it
  only runs on `google/gemma-2-2b`.
- **exp2 / exp3 / exp4** use **nnsight** (`src/obench_interp/activations.py`) and
  take a `--model <hf-id>` flag. nnsight wraps the HF model as-is (no
  TransformerLens weight-conversion, which is a ~2x transient RAM peak that OOMs
  a 24 GB box on a 2B model), and works on any `model.model.layers[i]` decoder —
  gemma-2, Qwen2.5/3, Llama-3, Mistral. `--layer` defaults to the model's middle
  layer. These need an **instruct** model that actually writes couplets / chains
  of thought / working code; a 0.5–2B base model produces none of them. exp2's
  SAE feature step only runs on `google/gemma-2-2b`; exp4 defaults to
  `Qwen/Qwen2.5-3B-Instruct`.

Confirmed model (2026-09-01): **`google/gemma-2-2b`**, layer 12 residual stream,
Gemma Scope `gemma-scope-2b-pt-res-canonical` width-16k SAE. See
`docs/model-choice.md` for the reasoning.

## Viewer (webui `#/interp`)

The chat webui (`tools/ui/`) has an **Interpretability** tab (flask icon) that
renders these runs — cosine heatmaps for exp1, per-couplet
baseline/patched/control panels for exp2, plain/pressured/ablated code with a
visible-vs-held-out score for exp4, side-by-side chains of thought with
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

## Live viewer (webui `#/interp` -> Live)

The batch `run exp*` path above works on a curated dataset. The **Live** view
instead lets you type a prompt and watch the interpretability signals stream in
per token. It needs a second sidecar that holds a loaded fp16 model:

```bash
./interp serve            # or  .\interp.ps1 serve   (loopback :8088)
```

This one is not read-only and not free of the model runtime — it runs the same
nnsight-wrapped HF model as `exp2`/`exp3`, so use the wrapper (venv + HF token)
rather than a bare `python`. `GET /models` lists the causal-LM repos already in
`hf_cache/` that it can load; `gemma-2-2b-it` and a small instruct model like
`Qwen/Qwen2.5-3B-Instruct` are the intended targets on an 11 GB card.

The Live tab has a panel per research question — see [`docs/live.md`](docs/live.md):

- **Q1 (language in its head)** — per-layer logit lens on every token, classified
  by script + function words, plus the Gemma Scope SAE features firing at the
  probe layer (cross-referenced with any `run exp1` output).
- **Q2 (planning ahead)** — once a sentence completes, how many tokens early its
  final word entered the model's next-token candidates; a **causal test** button
  runs exp2's activation patch on a prompt pair.
- **Q3 (CoT faithfulness)** — with a biasing context supplied, whether the
  reasoning acknowledges it; a **causal test** button runs exp3's hint ablation.
- **Q4 (specification gaming)** — give a coding task plus the visible test
  inputs; whether the reasoning describes an algorithm while the code just
  hardcodes the tests, plus a per-token surprisal trace. A **causal test**
  button runs exp4's pressure-frame ablation.

The live meters are heuristic previews; the causal-test buttons are the real
`exp2` / `exp3` / `exp4` activation-patching experiments on a single item.

Per-token generation runs the HF model directly with a KV cache (~9-10 tok/s for
a 2B model on the 1080 Ti). The per-layer logit lens is the main per-token cost —
raise `lens_stride` in the tab if it drags. VRAM is handed back to `llama-server`
after each turn.
