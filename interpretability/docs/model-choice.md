# Model choice for Phase 1 (multilingual concept sharing)

**Status: awaiting Josh's confirmation. Do not pull weights or install deps until settled.**

## Recommendation: `google/gemma-2-2b` (base, not -it)

### Why

- **Pretrained SAEs already exist**: DeepMind's *Gemma Scope* ships JumpReLU SAEs
  for every layer of gemma-2-2b, distributed through SAELens
  (`release="gemma-scope-2b-pt-res-canonical"`). The build plan says to check
  public SAE availability *before* committing to training one. For gemma-2-2b the
  answer is a clean yes, which removes the biggest risk from Phase 1.
- **TransformerLens support is first-class**: `HookedTransformer.from_pretrained("gemma-2-2b")`
  is a supported, tested path. Residual-stream hooks (`blocks.{L}.hook_resid_post`)
  line up directly with the Gemma Scope `res` SAEs.
- **Fits the 1080 Ti**: ~5.2 GB fp16 weights. Load with `device_map`/CPU-offload
  to keep the TransformerLens conversion peak under 11 GB; inference on a few
  dozen short prompts is fine. CPU fallback works if VRAM is tight.
- Multilingual coverage (en/fr/de/es + some zh) is adequate for an antonym task.
  The published Gemma Scope multilingual-feature demos use exactly this model.

### Suggested Phase 1 params

- Layer: **12** (of 26) to start, `blocks.12.hook_resid_post`, SAE width 16k.
  Sweep 6-20 if the middle layer is uninformative.
- SAE: `gemma-scope-2b-pt-res-canonical`, id `layer_12/width_16k/canonical`.

### Caveats to confirm

- Gemma weights are **gated on HuggingFace**. Josh needs to accept the license
  and have an `HF_TOKEN` with access. (Per repo convention, that token lives in
  `/var/secrets/nowservingto.env`, sourced by the script, never printed.)
- Disk: ~5 GB model + ~1-2 GB for the SAEs actually used.

## Alternative: `Qwen/Qwen2.5-1.5B`

Stronger multilingual model, ~3 GB fp16, not gated. **But**: no comprehensive
public SAE suite for Qwen2.5 that's wired into SAELens the way Gemma Scope is, so
Phase 1 would need to train an SAE with `sae_lens` (extra time + a real risk of
a weak SAE that muddies the cross-language overlap result). Recommend only if the
multilingual signal on gemma-2-2b turns out too weak.

## Not viable on this hardware

`Qwen/Qwen2.5-7B` / any 7B+ in fp16 (~15 GB) exceeds 11 GB VRAM.

## Decision

- [ ] gemma-2-2b + Gemma Scope (recommended)
- [ ] Qwen2.5-1.5B + train our own SAE
- [ ] something else: __________
