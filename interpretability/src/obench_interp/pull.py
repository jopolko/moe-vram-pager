"""Weight puller. Powers `obench-interp pull`. Wraps huggingface_hub.

HF token is read from the environment (HF_TOKEN etc.). Per repo convention it
lives in /var/secrets/nowservingto.env and is sourced by the wrapper script,
never printed here.
"""
from __future__ import annotations

import os
import sys

from .config import INSTRUCT, ModelConfig
from .env import HF_CACHE, _hf_token

os.environ.setdefault("HF_HOME", str(HF_CACHE))


def _sae_repo(cfg: ModelConfig) -> str:
    kind = "it" if cfg.sae_release and "-it-" in cfg.sae_release else "pt"
    return f"google/gemma-scope-2b-{kind}-res"


def pull(include_instruct: bool = False) -> None:
    from huggingface_hub import snapshot_download

    token = _hf_token()
    if not token:
        sys.exit(
            "No HF token in env. Source the secrets file first, e.g.:\n"
            "  set -a; source /var/secrets/nowservingto.env; set +a\n"
            "or use the wrapper: ./interp pull"
        )

    targets = [ModelConfig()]
    if include_instruct:
        targets.append(INSTRUCT)

    for cfg in targets:
        print(f"== {cfg.hf_name} (safetensors only) ==")
        snapshot_download(
            cfg.hf_name,
            allow_patterns=["*.safetensors", "*.json", "tokenizer.model"],
            token=token,
        )
        # Let sae_lens resolve the canonical release -> the right average_l0
        # subdir and fetch it. Hand-written HF patterns get this wrong because
        # "canonical" is a sae_lens alias, not a real directory.
        from sae_lens import SAE

        print(f"== SAE {cfg.sae_release} / {cfg.sae_id} ==")
        SAE.from_pretrained(cfg.sae_release, cfg.sae_id, device="cpu")

    print(f"\nDone. Cache: {HF_CACHE}")
