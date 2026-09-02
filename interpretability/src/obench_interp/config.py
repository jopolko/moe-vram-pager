"""Shared runtime config: device, dtype, model identity.

Hardware target: single GTX 1080 Ti (Pascal, 11 GB, no hardware bf16).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # Pascal has no hardware bf16. Use fp16 on GPU, fp32 on CPU.
    # fp16 here is for memory, not speed: Pascal fp16 compute is slow.
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:  # Ampere+ can do bf16
            return torch.bfloat16
        return torch.float16
    return torch.float32


@dataclass(frozen=True)
class ModelConfig:
    # Confirmed 2026-09-01. See docs/model-choice.md.
    hf_name: str = "google/gemma-2-2b"
    # TransformerLens model name (matches hf_name for gemma-2-2b).
    tl_name: str = "gemma-2-2b"
    # Residual-stream hook point for Phase 1: middle layer, 12 of 26.
    # Sweep 6-20 if layer 12 is uninformative.
    layer: int = 12
    hook_name: str = "blocks.12.hook_resid_post"
    # Gemma Scope: pretrained JumpReLU SAEs, no training needed.
    sae_release: str | None = "gemma-scope-2b-pt-res-canonical"
    sae_id: str | None = "layer_12/width_16k/canonical"


# Phase 3 (CoT faithfulness) swaps to the instruct model. It is pure activation
# patching - no SAE involved - and DeepMind never released a Gemma Scope residual
# SAE trained on gemma-2-2b-it (sae_lens only registers the -pt- set), so there is
# no SAE to pull or check here. If a future exp3 variant wants SAE feature
# attribution on the instruct model, the accepted route is the base -pt- SAEs
# applied to the instruct residual stream - set sae_release back to the pt release.
INSTRUCT = ModelConfig(
    hf_name="google/gemma-2-2b-it",
    tl_name="gemma-2-2b-it",
    layer=12,
    hook_name="blocks.12.hook_resid_post",
    sae_release=None,
    sae_id=None,
)


DEVICE = pick_device()
DTYPE = pick_dtype(DEVICE)
