#!/usr/bin/env python3
"""
Auto-tune ctx-size and KV cache quantization for models in the router's
models-preset.ini, using each GGUF's own metadata instead of a guessed
default (this is what caught JOSIEFIED-Qwen3:8b sitting at ctx-size=16384
when the model's real trained context is 40960 - see qwen3.context_length).

For each model: reads {arch}.context_length as the safe ctx-size ceiling
(no attempt to extrapolate past it via rope scaling - that trades quality
for a longer window and isn't a "free" win the way this is), computes what
q8_0 K/V cache quantization saves in KV cache memory using the model's own
attention geometry ({arch}.block_count / head_count_kv / key_length /
value_length), and pushes both through the router's existing
POST /models/tuning endpoint (writes the ini, sets need_reload) followed by
GET /models?reload=1 to apply it - the same mechanism the webui's own
tuning UI uses, so this doesn't bypass anything the router doesn't already
expect.

cache-type-k/v default to q8_0 rather than q4_0: q8_0 is the well-established
"basically free" choice (halves KV memory, negligible quality cost); q4_0
saves more but is a real quality tradeoff and shouldn't be applied blindly
across every model without the user asking for it specifically.

By default the KV-cache budget is auto-detected from the GPU (nvidia-smi
total VRAM) rather than always recommending the model's full native ctx_train
- native ctx_train on its own says nothing about whether the KV cache for it
actually fits *this* card. The auto budget subtracts the model's own weight
size (GGUF file size, a fair proxy for VRAM footprint when fully offloaded)
and its preset's own moe-stream-cache reservation (VRAM the streaming
expert-cache mechanism already claims) from total VRAM, then leaves a fixed
safety margin for CUDA context/other processes. This is what catches a case
like gemma-4-abliterated:latest, whose native ctx_train (131072) needs an
~11 GiB KV cache alone - more than an entire 11 GiB GTX 1080 Ti - which
--apply used to push through unconditionally and broke the model's ability
to load at all. Pass --vram-budget-gb to override the detected budget, or
--no-vram-budget to go back to always targeting the full native ctx_train.

Usage:
    tools/auto_kv_tune.py                       # dry run, all models, report only
    tools/auto_kv_tune.py --apply                # apply to all models, auto VRAM budget
    tools/auto_kv_tune.py --apply --model <name-or-alias-substring>
    tools/auto_kv_tune.py --apply --cache-type q4_0
    tools/auto_kv_tune.py --apply --vram-budget-gb 4   # override the detected budget
"""
import argparse
import configparser
import json
import re
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

DEFAULT_PRESET = Path.home() / ".cache/llama-moe-pager/models-preset.ini"
DEFAULT_ROUTER = "http://127.0.0.1:8080"
# Fixed headroom left un-budgeted for CUDA context overhead, compute buffers,
# and anything else sharing the card - not a tunable knob, just a margin of
# error since none of those are precisely knowable from here.
VRAM_SAFETY_MARGIN_BYTES = 1 * (1024 ** 3)

_SIZE_SUFFIX = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}

GGUF_MAGIC = 0x46554747
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

_SCALAR_FMT = {
    0: "B", 1: "b", 2: "H", 3: "h", GGUF_TYPE_UINT32: "I", GGUF_TYPE_INT32: "i",
    GGUF_TYPE_FLOAT32: "f", GGUF_TYPE_UINT64: "Q", GGUF_TYPE_INT64: "q",
    GGUF_TYPE_FLOAT64: "d", GGUF_TYPE_BOOL: "?",
}


def _read_gguf_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype: int):
    if vtype == GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    if vtype == GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    fmt = _SCALAR_FMT[vtype]
    size = struct.calcsize(fmt)
    (val,) = struct.unpack("<" + fmt, f.read(size))
    return val


def read_gguf_metadata(path: Path) -> dict:
    """Minimal GGUF header reader - just the KV metadata dict, no tensors.
    Avoids depending on gguf-py's tensor-loading path (which wants the whole
    file mapped) just to read a dozen scalar fields."""
    with open(path, "rb") as f:
        magic, version, _tensor_count, kv_count = struct.unpack("<IIQQ", f.read(24))
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (bad magic)")
        meta = {}
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            meta[key] = _read_gguf_value(f, vtype)
        return meta


def kv_bytes_per_token(meta: dict, arch: str, bytes_per_elem: float) -> float | None:
    block_count = meta.get(f"{arch}.block_count")
    head_kv = meta.get(f"{arch}.attention.head_count_kv")
    key_len = meta.get(f"{arch}.attention.key_length")
    val_len = meta.get(f"{arch}.attention.value_length")
    # key_length/value_length are optional in GGUF - llama.cpp itself falls back to
    # embedding_length / head_count (not head_count_kv) when they're absent, so do
    # the same instead of treating every model missing them as "unknown" (this is
    # most models - explicit key/value_length is the exception, not the rule).
    if key_len is None or val_len is None:
        embd = meta.get(f"{arch}.embedding_length")
        head_count = meta.get(f"{arch}.attention.head_count")
        if isinstance(embd, int) and isinstance(head_count, int) and head_count:
            derived = embd // head_count
            key_len = key_len if key_len is not None else derived
            val_len = val_len if val_len is not None else derived
    values = (block_count, head_kv, key_len, val_len)
    # Some architectures (hybrid/sliding-window attention, MoE variants) store
    # these per-layer as arrays instead of one scalar for the whole model -
    # not worth reconstructing the real per-layer total here, just skip the
    # KV-size estimate for those and still let ctx-size/cache-type tuning
    # proceed above.
    if any(v is None or isinstance(v, list) for v in values):
        return None
    return block_count * head_kv * (key_len + val_len) * bytes_per_elem


CACHE_TYPE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0625,   # 1 byte + block scale overhead (~1/32 elements)
    "q4_1": 0.625, "q4_0": 0.5625,
    "q5_0": 0.6875, "q5_1": 0.75, "iq4_nl": 0.5625,
}


def analyze_model(gguf_path: Path, cache_type: str) -> dict:
    meta = read_gguf_metadata(gguf_path)
    arch = meta.get("general.architecture")
    if not arch:
        raise ValueError(f"{gguf_path}: no general.architecture in metadata")

    ctx_train = meta.get(f"{arch}.context_length")
    scaling_type = meta.get(f"{arch}.rope.scaling.type")
    scaling_factor = meta.get(f"{arch}.rope.scaling.factor")
    extended = bool(scaling_type and scaling_type != "none" and scaling_factor)

    bpt_f16 = kv_bytes_per_token(meta, arch, CACHE_TYPE_BYTES["f16"])
    bpt_target = kv_bytes_per_token(meta, arch, CACHE_TYPE_BYTES.get(cache_type, 2.0))
    expert_count = meta.get(f"{arch}.expert_count")

    return {
        "arch": arch,
        "ctx_train": ctx_train,
        "rope_extended": extended,
        "kv_bytes_per_token_f16": bpt_f16,
        "kv_bytes_per_token_target": bpt_target,
        "is_moe": bool(expert_count),
    }


def resolve_model_path(section: dict, section_name: str) -> Path | None:
    model_path = section.get("model") or section_name
    p = Path(model_path)
    return p if p.exists() else None


def human_gib(n_bytes: float) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GiB"


def parse_size_to_bytes(value: str) -> int | None:
    """Parse sizes like the ones already used in models-preset.ini (e.g.
    moe-stream-cache = "2G") - bare digits are bytes, a trailing K/M/G/T
    (case-insensitive) scales accordingly."""
    if not value:
        return None
    m = re.match(r"^\s*([\d.]+)\s*([kmgtKMGT]?)\s*$", value)
    if not m:
        return None
    num, suffix = m.groups()
    mult = _SIZE_SUFFIX.get(suffix.lower(), 1)
    return int(float(num) * mult)


_NVIDIA_SMI_CANDIDATES = ["nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"]


def get_gpu_total_vram_bytes() -> int | None:
    """Total VRAM on the first NVIDIA GPU, via nvidia-smi. Returns None (not
    an error) when nvidia-smi isn't present or no GPU is found - callers
    should treat that as "budget unknown", not "budget is zero".

    Tries plain "nvidia-smi" first (works in an interactive shell where WSL's
    /usr/lib/wsl/lib is on PATH), then the WSL-specific absolute path - a
    systemd service unit's minimal PATH doesn't include that directory, and
    silently losing VRAM detection there is exactly what let ctx-size get
    pushed back up to unfittable values via auto-kv-tune.service."""
    for candidate in _NVIDIA_SMI_CANDIDATES:
        try:
            out = subprocess.run(
                [candidate, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        break
    else:
        return None
    first_line = out.splitlines()[0].strip() if out else ""
    if not first_line.isdigit():
        return None
    return int(first_line) * (1024 ** 2)  # nvidia-smi reports MiB


def compute_auto_vram_budget_bytes(
        total_vram_bytes: int, gguf_path: Path, section: dict, is_moe: bool) -> int:
    """VRAM left over for KV cache after the model's own weight footprint and
    a fixed safety margin for CUDA context/other processes.

    Weight footprint depends entirely on whether this model is dense or MoE:
    - Dense: the whole GGUF file ends up resident in VRAM when fully offloaded,
      so its on-disk size is a fair proxy. moe-stream-cache is NOT an
      additional allocation here: llama_moe_stream_resolve_slots() (see
      src/llama-model.cpp) returns 0 slots whenever n_expert == 0, so the
      streaming cache object is never constructed and alloc_bufs() never
      runs - the flag is accepted but silently a no-op for a dense model, it
      does not reserve any VRAM. Subtracting it here would double-count VRAM
      that was never actually spent, artificially shrinking ctx-size for no
      reason (this is exactly what made qwen2.5-coder-14b get capped at
      ctx-size=6467 instead of ~16748 on an 11 GiB card).
    - MoE under --moe-stream: the entire point of streaming is that most
      expert weights live in RAM/disk, not VRAM - only a bounded working set
      is GPU-resident at a time, sized by moe-stream-cache itself. Using the
      full file size here would treat all that freed-up VRAM as unavailable,
      defeating the actual feature this fork exists for. moe-stream-cache IS
      the weight-VRAM budget in this case, not an addition on top of it.
    """
    if is_moe:
        weight_vram_bytes = parse_size_to_bytes(section.get("moe-stream-cache", "")) or 0
    else:
        weight_vram_bytes = gguf_path.stat().st_size
    return total_vram_bytes - weight_vram_bytes - VRAM_SAFETY_MARGIN_BYTES


def router_post(router: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{router}{path}", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def router_get(router: str, path: str) -> None:
    with urllib.request.urlopen(f"{router}{path}", timeout=30):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    ap.add_argument("--router", default=DEFAULT_ROUTER)
    ap.add_argument("--model", help="Only touch sections whose name/alias contains this substring")
    ap.add_argument("--cache-type", default="q8_0", choices=sorted(CACHE_TYPE_BYTES))
    ap.add_argument("--apply", action="store_true", help="Actually push changes via /models/tuning (default: dry run)")
    ap.add_argument("--skip-ctx", action="store_true", help="Only tune cache-type-k/v, leave ctx-size alone")
    ap.add_argument("--vram-budget-gb", type=float,
                     help="Cap ctx-size per model so its KV cache (at --cache-type) fits this many GiB. "
                          "Overrides auto-detection - use this when nvidia-smi isn't available/accurate, or "
                          "you want a deliberately smaller number (e.g. to leave room for other models loaded "
                          "at once). Default: auto-detected per model from GPU total VRAM minus that model's own "
                          "weight footprint (dense: full file size; MoE: moe-stream-cache, see module docstring) "
                          "and a 1 GiB safety margin.")
    ap.add_argument("--no-vram-budget", action="store_true",
                     help="Disable VRAM budgeting entirely and always target the model's full native ctx_train, "
                          "the old (unsafe) default - only for when you're deliberately going to compensate some "
                          "other way (e.g. reducing -ngl yourself).")
    ap.add_argument("--min-ctx", type=int, default=2048,
                     help="Refuse to shrink a model below this ctx-size even if the VRAM budget would otherwise "
                          "push it lower (default: 2048) - below this a model stops being usable for most real work")
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # preserve key case
    cfg.read(args.preset)

    auto_vram_bytes = None
    vram_detection_failed = False
    if not args.no_vram_budget and args.vram_budget_gb is None:
        auto_vram_bytes = get_gpu_total_vram_bytes()
        if auto_vram_bytes is None:
            vram_detection_failed = True
            print("WARNING: nvidia-smi unavailable - can't auto-detect VRAM. Refusing to touch "
                  "ctx-size on any model this run (leaving cache-type-k/v tuning unaffected). "
                  "Pass --vram-budget-gb to set a budget manually, or --no-vram-budget to "
                  "deliberately go back to always targeting full native ctx_train.\n")

    for section_name in cfg.sections():
        section = dict(cfg[section_name])
        alias = section.get("alias", "")
        if args.model and args.model not in section_name and args.model not in alias:
            continue

        gguf_path = resolve_model_path(section, section_name)
        label = alias or section_name
        if gguf_path is None:
            print(f"[skip] {label}: model file not found on disk (not resolvable, may be a HF-id-only entry)")
            continue

        try:
            info = analyze_model(gguf_path, args.cache_type)
        except (ValueError, OSError, struct.error) as exc:
            print(f"[skip] {label}: couldn't read GGUF metadata ({exc})")
            continue

        current_ctx = int(section.get("ctx-size", 0) or 0)
        ctx_train = info["ctx_train"]
        current_cache_k = section.get("cache-type-k", "f16 (default)")
        bpt_target = info["kv_bytes_per_token_target"]

        print(f"\n{label}")
        print(f"  arch={info['arch']}  native ctx_train={ctx_train}  "
              f"rope-extended={info['rope_extended']}  is_moe={info['is_moe']}")
        print(f"  current: ctx-size={current_ctx or '(unset)'}  cache-type-k={current_cache_k}")

        # Resolve this model's KV budget: an explicit --vram-budget-gb always wins: it's a
        # deliberate KV-only cap regardless of weight footprint. Otherwise, when auto-detection
        # is on and succeeded, back out a per-model budget that accounts for what this specific
        # model's weights actually cost in VRAM (dense vs MoE-streamed - see
        # compute_auto_vram_budget_bytes).
        budget_gb = args.vram_budget_gb
        if budget_gb is None and auto_vram_bytes is not None:
            per_model_budget_bytes = compute_auto_vram_budget_bytes(
                auto_vram_bytes, gguf_path, section, info["is_moe"])
            budget_gb = per_model_budget_bytes / (1024 ** 3)
            print(f"  auto VRAM budget: {human_gib(auto_vram_bytes)} total - weight footprint "
                  f"({'moe-stream-cache' if info['is_moe'] else 'full file size'}) "
                  f"- {human_gib(VRAM_SAFETY_MARGIN_BYTES)} margin = {budget_gb:.2f} GiB for KV cache")

        desired_ctx = ctx_train
        if vram_detection_failed:
            desired_ctx = current_ctx or None
        elif budget_gb is not None and ctx_train:
            if bpt_target is None:
                print(f"  WARNING: can't estimate KV cache size for arch={info['arch']} "
                      f"(non-scalar attention geometry) - leaving ctx-size untouched rather than "
                      f"guessing at a budget-safe value. Use --skip-ctx or tune this one manually.")
                desired_ctx = current_ctx or None
            elif budget_gb <= 0:
                print(f"  WARNING: auto VRAM budget is {budget_gb:.2f} GiB (<=0) - this model's own "
                      f"weight footprint alone may not fit on this GPU. Leaving ctx-size untouched.")
                desired_ctx = current_ctx or None
            else:
                budget_bytes = budget_gb * (1024 ** 3)
                max_ctx_for_budget = int(budget_bytes // bpt_target)
                if max_ctx_for_budget < ctx_train:
                    if max_ctx_for_budget < args.min_ctx:
                        print(f"  WARNING: budget only fits ctx-size={max_ctx_for_budget} at "
                              f"{args.cache_type}, below --min-ctx={args.min_ctx} - leaving "
                              f"ctx-size untouched, this budget is too small for this model.")
                        desired_ctx = current_ctx or None
                    else:
                        desired_ctx = max_ctx_for_budget
                        print(f"  budget cap: {ctx_train} -> {desired_ctx} to fit "
                              f"{budget_gb:.2f} GiB of KV cache at {args.cache_type}")

        overrides = {}
        if not args.skip_ctx and desired_ctx and current_ctx != desired_ctx:
            overrides["ctx-size"] = str(desired_ctx)
            print(f"  -> ctx-size {current_ctx or '(unset)'} -> {desired_ctx}")

        if current_cache_k != args.cache_type:
            overrides["cache-type-k"] = args.cache_type
            overrides["cache-type-v"] = args.cache_type
            if info["kv_bytes_per_token_f16"] and bpt_target:
                use_ctx = desired_ctx if not args.skip_ctx and desired_ctx else current_ctx
                if use_ctx:
                    before = human_gib(info["kv_bytes_per_token_f16"] * use_ctx)
                    after = human_gib(bpt_target * use_ctx)
                    print(f"  -> cache-type-k/v f16 -> {args.cache_type}  "
                          f"(KV cache at ctx={use_ctx}: {before} -> {after})")

        if not overrides:
            print("  (already tuned, nothing to do)")
            continue

        if not args.apply:
            print("  (dry run - pass --apply to push this via /models/tuning)")
            continue

        try:
            router_post(args.router, "/models/tuning",
                        {"model": section_name, "overrides": overrides})
            router_get(args.router, "/models?reload=1")
            print("  applied.")
        except Exception as exc:  # noqa: BLE001 - report and continue with other models
            print(f"  ERROR applying via router ({args.router}): {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
