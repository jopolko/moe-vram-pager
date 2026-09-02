"""Environment / readiness checks. Powers `obench-interp doctor`.

Everything here is read-only and safe to run before weights exist.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import DEVICE, DTYPE, INSTRUCT, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERP_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = INTERP_ROOT / "hf_cache"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    return None


def _model_present(cfg: ModelConfig) -> bool:
    slug = "models--" + cfg.hf_name.replace("/", "--")
    snap = HF_CACHE / "hub" / slug / "snapshots"
    if not snap.is_dir():
        return False
    return any(p.suffix == ".safetensors" for p in snap.rglob("*.safetensors"))


def _sae_present(cfg: ModelConfig) -> bool:
    if not cfg.sae_release:
        return False
    # Gemma Scope canonical repos resolve to google/gemma-scope-2b-{pt,it}-res.
    # sae_lens picks a specific average_l0_* subdir; just look for the layer's npz.
    kind = "it" if "-it-" in cfg.sae_release else "pt"
    slug = f"models--google--gemma-scope-2b-{kind}-res"
    snap = HF_CACHE / "hub" / slug / "snapshots"
    if not snap.is_dir():
        return False
    layer_dir = cfg.sae_id.split("/")[0]  # e.g. "layer_12"
    # as_posix() so the "/layer_12/" match works on Windows too (str(p) would use "\")
    return any(f"/{layer_dir}/" in p.as_posix() for p in snap.rglob("*.npz"))


def gpu_report() -> Check:
    try:
        import torch
    except ImportError:
        return Check("gpu", False, "torch not importable (run: uv pip install -e .)")
    if not torch.cuda.is_available():
        return Check("gpu", True, "no CUDA device; will run on CPU (slow but works for 2B)")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    gb = 1024**3
    major, _ = torch.cuda.get_device_capability()
    note = "" if major >= 8 else " (Pascal/no bf16 -> fp16)"
    return Check("gpu", True, f"{name}{note}, {free / gb:.1f} GB free / {total / gb:.1f} GB")


def disk_report() -> Check:
    usage = shutil.disk_usage(INTERP_ROOT)
    gb = 1024**3
    free = usage.free / gb
    return Check("disk", free > 15, f"{free:.0f} GB free at {INTERP_ROOT} (need ~10 GB for weights+SAEs)")


def run_all() -> list[Check]:
    checks: list[Check] = []

    try:
        import torch  # noqa: F401
        import transformer_lens  # noqa: F401
        import sae_lens  # noqa: F401
        import nnsight  # noqa: F401

        from importlib.metadata import version

        vs = ", ".join(f"{p} {version(p)}" for p in ("torch", "transformer_lens", "sae_lens", "nnsight"))
        checks.append(Check("deps", True, vs))
    except ImportError as e:
        checks.append(Check("deps", False, f"missing: {e.name}. Run: uv pip install -e ."))

    checks.append(gpu_report())
    checks.append(disk_report())

    tok = _hf_token()
    checks.append(
        Check("hf_token", bool(tok), "found in env" if tok else "not set (needed only for `pull`)")
    )

    for label, cfg in (("phase-1 base", ModelConfig()), ("phase-3 instruct", INSTRUCT)):
        m = _model_present(cfg)
        s = _sae_present(cfg)
        checks.append(
            Check(
                f"weights ({label})",
                m and s,
                f"{cfg.hf_name}: model={'yes' if m else 'MISSING'}, "
                f"SAE {cfg.sae_id}={'yes' if s else 'MISSING'}"
                + ("" if (m and s) else "  -> run: obench-interp pull"),
            )
        )

    checks.append(Check("device", True, f"selected: {DEVICE} / {str(DTYPE).replace('torch.', '')}"))
    return checks
