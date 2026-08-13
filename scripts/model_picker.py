#!/usr/bin/env python3
"""
Cross-reference the UGI-Leaderboard against this hardware and this fork's
conversion support, and rank MoE models by whether --moe-stream can actually
make them usable.

The fit heuristic: for a MoE model, what matters isn't total size (disk is
cheap and --moe-stream keeps cold experts off it), it's the active-params
working set that has to stay cache-resident every forward pass. That set is
compared against --vram-gb (the eventual VRAM cache tier) and --ram-gb (the
RAM cache tier that exists today).

Data sources:
- UGI-Leaderboard CSV (author/model_name, Active/Total Parameters, Architecture,
  UGI score, willingness score): https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard
- This fork's own conversion/__init__.py TEXT_MODEL_MAP, which is the ground
  truth for "can convert_hf_to_gguf.py actually convert this architecture".
- HF's model search API, used only for the top-N results, to check whether a
  community GGUF quant already exists (bartowski/mradermacher/etc.) instead
  of needing the full bf16 checkpoint.
- derestricted-filter.json in this repo, fetched fresh (cached ~1 day) so
  new terms/models added on GitHub reach every client without a rebuild.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

UGI_CSV_URL = "https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard/resolve/main/ugi-leaderboard-data.csv"
CACHE_PATH = Path(__file__).parent / ".cache" / "ugi-leaderboard-data.csv"
CACHE_TTL_SECONDS = 24 * 3600

TAGS_CACHE_PATH = Path(__file__).parent / ".cache" / "hf-tags.json"
TAGS_CACHE_TTL_SECONDS = 14 * 24 * 3600

# Public repo, public file: no auth needed on either WSL or the Windows-native
# client. Edit derestricted-filter.json on GitHub and every client picks it
# up within a day (or immediately with --refresh), no rebuild/resync needed.
DERESTRICTED_FILTER_URL = "https://raw.githubusercontent.com/jopolko/moe-vram-pager/main/derestricted-filter.json"
DERESTRICTED_FILTER_CACHE_PATH = Path(__file__).parent / ".cache" / "derestricted-filter.json"
DERESTRICTED_FILTER_TTL_SECONDS = 24 * 3600

REPO_ROOT = Path(__file__).parent.parent
CONVERSION_INIT = REPO_ROOT / "conversion" / "__init__.py"

# Approximate bits-per-weight for llama.cpp quant types, used to estimate
# file/working-set size from a raw parameter count. Not exact (k-quants vary
# slightly by tensor and model), close enough for a go/no-go estimate.
QUANT_BPW = {
    "Q4_K_S": 4.58,
    "Q4_K_M": 4.83,
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.67,
    "Q6_K": 6.56,
    "Q8_0": 8.50,
    "IQ4_XS": 4.25,
    "IQ3_M": 3.66,
}

# Built-in fallback, used if derestricted-filter.json can't be fetched and
# there's no local cache yet. Matched against both the model name and (more
# reliably) its HF tags list. "heretic" is a known automated-abliteration
# tool whose output repos tag themselves with it; found by checking an
# actual model page rather than guessing, same for nsfw/not-for-all-audiences
# which are real HF tags. The second row was added after cross-checking real
# HF tags pulled for ~160 candidate repos: each term was confirmed on an
# actual model page, not guessed ("jailbreak" is the one borderline case -
# seen once, alongside "red-teaming"/"evaluation" tags, so it may also catch
# a red-team eval repo rather than a ready-to-use uncensored finetune).
DEFAULT_DERESTRICTED_TERMS = [
    "abliterat", "derestrict", "uncensor", "decensor", "unalign",
    "unshackl", "unfilter", "unbound", "unrestrict", "heretic",
    "nsfw", "not-for-all-audiences", "de-alignment",
    "obliterat", "ablated", "unlimited", "refusal-remov", "amoral", "jailbreak",
]
# Specific repo ids to always flag regardless of name/tag matching, for
# cases where a model doesn't self-tag accurately. Add these on GitHub in
# derestricted-filter.json, not here.
DEFAULT_KNOWN_DERESTRICTED_REPOS: list[str] = []


def clean_repo_id(name: str) -> str:
    """Strip UGI-Leaderboard display annotations like " (thinking=True)".

    These aren't part of the actual HF repo id (repo ids can't contain
    spaces or parens), so leaving them in makes the tags/GGUF API lookups
    404 silently.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


@dataclass
class ModelRow:
    name: str
    link: str
    active_b: float
    total_b: float
    hf_arch: str
    ugi_score: float
    willingness: float
    is_finetuned: bool

    repo_id: str = ""
    gguf_arch: str | None = None
    fit_tier: str = ""
    active_gb: float = 0.0
    total_gb: float = 0.0
    is_derestricted: bool = False
    gguf_repo: str | None = None


def load_arch_map() -> dict[str, str]:
    """Parse TEXT_MODEL_MAP straight out of conversion/__init__.py.

    This is the authoritative "does this fork's convert script recognize
    this HF architecture" list, kept in sync automatically since it's read
    from source rather than hand-copied.
    """
    src = CONVERSION_INIT.read_text()
    m = re.search(r"TEXT_MODEL_MAP:\s*dict\[str,\s*str\]\s*=\s*(\{.*?\n\})", src, re.S)
    if not m:
        raise RuntimeError(f"could not find TEXT_MODEL_MAP in {CONVERSION_INIT}")
    return ast.literal_eval(m.group(1))


def load_ugi_csv(refresh: bool) -> list[dict]:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not refresh and CACHE_PATH.exists() and (time.time() - CACHE_PATH.stat().st_mtime) < CACHE_TTL_SECONDS:
        text = CACHE_PATH.read_text(encoding="utf-8-sig")
    else:
        resp = requests.get(UGI_CSV_URL, timeout=30)
        resp.raise_for_status()
        text = resp.content.decode("utf-8-sig")
        CACHE_PATH.write_text(text, encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def text_matches_derestricted(terms: list[str], *texts: str) -> bool:
    return any(kw in t.lower() for t in texts for kw in terms)


def load_derestricted_filter(refresh: bool) -> tuple[list[str], set[str]]:
    """Pull the community-maintained term/model list from GitHub, merged
    (additive only) with the built-in defaults so a bad or unreachable
    fetch never loses coverage, only misses out on additions.
    """
    remote_terms: list[str] = []
    remote_repos: list[str] = []

    use_cache = (
        not refresh
        and DERESTRICTED_FILTER_CACHE_PATH.exists()
        and (time.time() - DERESTRICTED_FILTER_CACHE_PATH.stat().st_mtime) < DERESTRICTED_FILTER_TTL_SECONDS
    )
    if use_cache:
        try:
            data = json.loads(DERESTRICTED_FILTER_CACHE_PATH.read_text())
            remote_terms = data.get("terms", [])
            remote_repos = data.get("known_repo_ids", [])
        except (json.JSONDecodeError, OSError):
            pass
    else:
        try:
            resp = requests.get(DERESTRICTED_FILTER_URL, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            remote_terms = data.get("terms", [])
            remote_repos = data.get("known_repo_ids", [])
            DERESTRICTED_FILTER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            DERESTRICTED_FILTER_CACHE_PATH.write_text(json.dumps(data))
        except (requests.RequestException, ValueError, OSError):
            # Offline, repo unreachable, whatever, fall back to whatever's
            # cached even if stale, then to the built-in list.
            if DERESTRICTED_FILTER_CACHE_PATH.exists():
                try:
                    data = json.loads(DERESTRICTED_FILTER_CACHE_PATH.read_text())
                    remote_terms = data.get("terms", [])
                    remote_repos = data.get("known_repo_ids", [])
                except (json.JSONDecodeError, OSError):
                    pass

    terms = sorted(set(DEFAULT_DERESTRICTED_TERMS) | set(remote_terms))
    known_repos = set(DEFAULT_KNOWN_DERESTRICTED_REPOS) | set(remote_repos)
    return terms, known_repos


def load_tags_cache() -> dict[str, list[str]]:
    if TAGS_CACHE_PATH.exists() and (time.time() - TAGS_CACHE_PATH.stat().st_mtime) < TAGS_CACHE_TTL_SECONDS:
        try:
            return json.loads(TAGS_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_tags_cache(cache: dict[str, list[str]]) -> None:
    TAGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TAGS_CACHE_PATH.write_text(json.dumps(cache))


def fetch_tags(model_id: str, timeout: float = 6.0) -> list[str]:
    try:
        resp = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            params={"expand[]": "tags"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("tags", [])
    except (requests.RequestException, ValueError):
        return []


def fetch_all_tags(model_ids: list[str], max_workers: int = 16) -> dict[str, list[str]]:
    """Fetch HF tags for every candidate, cached across runs (tags rarely change).

    This is what actually catches things a name-keyword match misses, e.g.
    a repo named after an abliteration tool ("Heretic-...") rather than the
    word "abliterated" itself, the tool tags its own output.
    """
    cache = load_tags_cache()
    to_fetch = [mid for mid in model_ids if mid not in cache]
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_tags, mid): mid for mid in to_fetch}
            for fut in as_completed(futures):
                mid = futures[fut]
                cache[mid] = fut.result()
        save_tags_cache(cache)
    return cache


def real_disk_free_gb(path: str) -> float:
    """Free space, capped by the real host drive's free space on WSL2.

    On WSL2, `path` typically lives inside a dynamically-growing virtual disk
    (a VHDX on the Windows host). Its self-reported free space is unallocated
    blocks *within that virtual disk*, which can be far larger than what's
    actually left on the physical host drive it still needs to grow into.
    /mnt/c (when present) is the real host C: drive via DrvFs, so cap by
    whichever is smaller. Assumes the common default single-drive (C:) WSL
    setup; won't catch a WSL install relocated to another drive letter.
    """
    free_gb = shutil.disk_usage(path).free / 1e9
    try:
        is_wsl = "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        is_wsl = False
    if is_wsl and Path("/mnt/c").is_dir():
        try:
            host_free_gb = shutil.disk_usage("/mnt/c").free / 1e9
            free_gb = min(free_gb, host_free_gb)
        except OSError:
            pass
    return free_gb


def classify_fit(active_gb: float, total_gb: float, vram_gb: float, ram_gb: float, disk_free_gb: float) -> str:
    """--moe-stream serves the model from disk with a bounded RAM-LRU cache in
    front of it, not "everything must be simultaneously resident." So the only
    hard failure is not being able to store the model at all (disk); a working
    set bigger than VRAM+RAM just means more cache misses streamed from disk
    per token (slower), not "won't run." Tiers below are purely informational
    (expected speed), never exclude a model.
    """
    # Reserve headroom: KV cache + OS + non-expert weights aren't counted
    # above, so don't pretend the full budget is free for the active set.
    easy_budget = vram_gb * 0.35
    vram_budget = vram_gb * 0.75
    ram_budget = ram_gb * 0.7

    if total_gb > disk_free_gb:
        return "no-disk-space"  # can't even be stored
    if active_gb <= easy_budget:
        return "easy"  # VRAM-resident, plenty of headroom
    if active_gb <= vram_budget:
        return "comfortable"  # VRAM-resident
    if active_gb <= vram_gb + ram_budget:
        return "ram-cache"  # spills into RAM cache, no disk streaming needed for the hot set
    return "disk-streaming"  # active set exceeds VRAM+RAM, expect real per-token disk streaming


# Used only to group results (fastest-expected first, then by UGI within a
# group) so the top-N selection isn't dominated by a faster-but-lower-UGI
# pick over a slower-but-much-better one; the tier distinction is
# informational (shown via fit_tier), not a ranking exclusion - nothing
# below is ever dropped from results except "no-disk-space".
FIT_ORDER = {"easy": 0, "comfortable": 0, "ram-cache": 1, "disk-streaming": 2, "no-disk-space": 3}

UNUSABLE_TIERS = {"no-disk-space"}


def gguf_search(model_name: str, timeout: float = 8.0) -> str | None:
    """Best-effort search for an existing community GGUF quant of model_name."""
    base = model_name.split("/")[-1]
    try:
        resp = requests.get(
            "https://huggingface.co/api/models",
            params={"search": f"{base} GGUF", "limit": 15},
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json()
    except (requests.RequestException, ValueError):
        return None

    base_lower = base.lower()
    for r in results:
        rid = r.get("id", "")
        if "gguf" in rid.lower() and base_lower.split("-derestricted")[0][:12] in rid.lower():
            return rid
    for r in results:
        rid = r.get("id", "")
        if "gguf" in rid.lower():
            return rid
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vram-gb", type=float, default=11.0, help="usable VRAM budget (default: 1080 Ti, 11GB)")
    ap.add_argument("--ram-gb", type=float, default=24.0, help="total system RAM (default: 24GB per this box)")
    ap.add_argument("--disk-path", type=str, default=str(REPO_ROOT), help="path to check free disk space on")
    ap.add_argument("--quant", choices=sorted(QUANT_BPW), default="Q4_K_M")
    ap.add_argument("--derestricted-only", action="store_true", help="only show derestricted/abliterated-tagged models")
    ap.add_argument("--min-ugi", type=float, default=0.0, help="minimum UGI score")
    ap.add_argument(
        "--rank-by", choices=("ugi", "size", "willingness"), default="ugi",
        help="ugi = best overall quality (default); size = biggest active-param model that still fits; "
             "willingness = least likely to refuse, regardless of general capability",
    )
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--gguf-lookup", type=int, default=10, help="check GGUF availability for the top N results (HF API calls, slow)")
    ap.add_argument("--refresh", action="store_true", help="force re-download the UGI CSV and derestricted-filter.json")
    ap.add_argument("--unsupported-only", action="store_true", help="show only architectures this fork can't convert yet (for gap-finding)")
    args = ap.parse_args()

    bpw = QUANT_BPW[args.quant]
    disk_free_gb = real_disk_free_gb(args.disk_path)

    arch_map = load_arch_map()
    rows = load_ugi_csv(args.refresh)

    models: list[ModelRow] = []
    for r in rows:
        active_b = to_float(r.get("Active Parameters"))
        total_b = to_float(r.get("Total Parameters"))
        if active_b <= 0 or total_b <= 0 or active_b >= total_b:
            continue  # not MoE, or missing data

        hf_arch = (r.get("Architecture") or "").strip()
        m = ModelRow(
            name=r.get("author/model_name", "").strip(),
            link=r.get("Model Link", "").strip(),
            active_b=active_b,
            total_b=total_b,
            hf_arch=hf_arch,
            ugi_score=to_float(r.get("UGI 🏆")),
            willingness=to_float(r.get("W/10 👍")),
            is_finetuned=(r.get("Is Finetuned", "").strip().upper() == "TRUE"),
        )
        m.repo_id = clean_repo_id(m.name)
        m.gguf_arch = arch_map.get(hf_arch)
        m.active_gb = active_b * 1e9 * bpw / 8 / 1e9
        m.total_gb = total_b * 1e9 * bpw / 8 / 1e9
        m.fit_tier = classify_fit(m.active_gb, m.total_gb, args.vram_gb, args.ram_gb, disk_free_gb)
        models.append(m)

    # The UGI CSV leaves Architecture blank for some (often very new) models -
    # that means "unknown," not "confirmed this fork can't load it." Default
    # view: include confirmed-supported and unknown, exclude only confirmed-
    # unsupported. unsupported_only (gap-finding): show only confirmed gaps,
    # not unknowns.
    if args.unsupported_only:
        models = [m for m in models if m.hf_arch and m.gguf_arch is None]
    else:
        models = [m for m in models if not m.hf_arch or m.gguf_arch is not None]

    # Tag lookup is what actually catches abliteration-tool-branded repos
    # (e.g. "Heretic-...") that a name-keyword match alone would miss, so
    # it runs on the whole filtered candidate set, not just the top N.
    terms, known_repos = load_derestricted_filter(args.refresh)
    tags_by_id = fetch_all_tags([m.repo_id for m in models])
    for m in models:
        m.is_derestricted = (
            m.repo_id in known_repos
            or text_matches_derestricted(terms, m.name, " ".join(tags_by_id.get(m.repo_id, [])))
        )

    if args.derestricted_only:
        models = [m for m in models if m.is_derestricted]

    models = [m for m in models if m.ugi_score >= args.min_ugi]
    models = [m for m in models if m.fit_tier not in UNUSABLE_TIERS]  # won't run on this hardware at all

    rank_keys = {
        "size": lambda m: -m.active_gb,
        "willingness": lambda m: -m.willingness,
        "ugi": lambda m: -m.ugi_score,
    }
    rank_key = rank_keys[args.rank_by]
    models.sort(key=lambda m: (FIT_ORDER.get(m.fit_tier, 9), rank_key(m)))
    models = models[: args.top]

    for i, m in enumerate(models):
        if i >= args.gguf_lookup:
            break
        m.gguf_repo = gguf_search(m.repo_id)

    print(f"# hardware: {args.vram_gb:.0f}GB VRAM, {args.ram_gb:.0f}GB RAM, {disk_free_gb:.0f}GB free disk, quant={args.quant}")
    print(f"# {len(models)} models shown (derestricted-only={args.derestricted_only}, min-ugi={args.min_ugi})")
    print()
    header = f"{'FIT':<13} {'ACTIVE':>7} {'TOTAL':>7} {'UGI':>6} {'W/10':>5} {'D':<2} {'MODEL':<55} GGUF"
    print(header)
    print("-" * len(header))
    for m in models:
        d_flag = "Y" if m.is_derestricted else ""
        gguf = m.gguf_repo or ("?" if m.gguf_repo is None else "none found")
        print(
            f"{m.fit_tier:<13} {m.active_gb:>6.1f}G {m.total_gb:>6.1f}G {m.ugi_score:>6.1f} "
            f"{m.willingness:>5.1f} {d_flag:<2} {m.name:<55} {gguf}"
        )


if __name__ == "__main__":
    sys.exit(main())
