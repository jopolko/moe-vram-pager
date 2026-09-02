"""Experiment 2: does the model plan multi-token output ahead of time?

Method
------
Rhyming couplets. Each item in `data/rhyming_couplets.json` has a `line1_clean`
that sets up rhyme family A and a length-matched `line1_corrupt` that sets up
rhyme family B. The model is prompted with `line1 + "\\n"` and must generate the
second line, which should end in a word from the matching family.

1. Load gemma-2-2b (TransformerLens), hook the residual stream at
   `blocks.{L}.hook_resid_post`.
2. Cache the residual stream for both the clean and corrupt prompts. The
   "planning position" is the last prompt token (the newline slot line 2 is
   generated from) -- at this point none of line 2 exists yet.
3. **Observational**: at the planning position, read off the model's own next-
   token probability for each family's `probe_words`. If family-A words are
   already favored over family-B words before any of line 2 has been written,
   the model has *already chosen* how the line will end.
4. **SAE**: encode the clean planning-position activation through the Gemma
   Scope SAE, record the top firing feature ids (for a future viz; this step
   makes no causal claim on its own).
5. **Causal**: generate line 2 from the clean prompt three ways --
   baseline (no intervention), patched (splice the corrupt run's residual
   stream into the clean run at the planning position, then generate), and
   control (same splice, but at an early line-1 position instead). If patching
   the planning position flips the generated line's ending from family A to
   family B far more often than the early-position control does, that is
   causal evidence the model was already planning the line ending at that
   position, not just biased by surface word statistics.

Output: results.json (shared envelope, see docs/result-schema.md) + summary.md,
under results/exp2_planning/.
"""
from __future__ import annotations

import argparse
import json
import re

from ..config import ModelConfig
from ..env import INTERP_ROOT
from ..loader import load_model, load_sae
from ..patching import cache_resid, generate_greedy, prob_of_words
from ..report import emit, new_run_dir

DATA = INTERP_ROOT / "data" / "rhyming_couplets.json"

MAX_NEW_TOKENS = 16


def _classify_rhyme(text: str, rhyme_a: list[str], rhyme_b: list[str]) -> str:
    """Classify a generated line's ending against the two candidate families.

    Matches on suffix of the normalized text (handles multi-word candidates
    like "grab it") or on the bare last word (handles single-word candidates
    like "rabbit"). Returns "a", "b", "ambiguous", or "other".
    """
    norm = re.sub(r"[^\w\s']", "", text.lower()).strip()
    words = norm.split()
    last = words[-1] if words else ""

    def hits(cands: list[str]) -> bool:
        for c in cands:
            c = c.lower().strip()
            if not c:
                continue
            # Multi-word candidates ("grab it") only match by suffix -- matching
            # on their bare last word ("it") would hit almost anything.
            if norm.endswith(c) or (" " not in c and last == c):
                return True
        return False

    a, b = hits(rhyme_a), hits(rhyme_b)
    if a and b:
        return "ambiguous"
    if a:
        return "a"
    if b:
        return "b"
    return "other"


def run(args: argparse.Namespace) -> dict:
    cfg = ModelConfig(layer=args.layer, hook_name=f"blocks.{args.layer}.hook_resid_post")
    items = json.loads(DATA.read_text())["items"]
    top_k = args.top_k

    print(f"[exp2] loading {cfg.hf_name} ...")
    model = load_model(cfg)
    print(f"[exp2] loading SAE {cfg.sae_release} / {cfg.sae_id} ...")
    sae = load_sae(cfg)

    per_item: list[dict] = []
    for item in items:
        clean_prompt = item["line1_clean"] + "\n"
        corrupt_prompt = item["line1_corrupt"] + "\n"
        rhyme_a, rhyme_b, probe_words = item["rhyme_a"], item["rhyme_b"], item["probe_words"]

        tok_clean = model.to_tokens(clean_prompt)
        tok_corrupt = model.to_tokens(corrupt_prompt)
        length_matched = tok_clean.shape[1] == tok_corrupt.shape[1]
        plan_pos = min(tok_clean.shape[1], tok_corrupt.shape[1]) - 1
        early_pos = max(1, plan_pos // 3)

        clean_logits, clean_resid = cache_resid(model, tok_clean, cfg.hook_name)
        _, corrupt_resid = cache_resid(model, tok_corrupt, cfg.hook_name)

        probe_probs = prob_of_words(model, clean_logits[0, plan_pos], probe_words)

        sae_acts = sae.encode(clean_resid[0, plan_pos].to(sae.W_enc.dtype)).float()
        topv, topi = sae_acts.topk(top_k)
        top_features = [
            {"id": int(i), "act": round(float(v), 4)} for v, i in zip(topv, topi) if v > 0
        ]

        baseline_text = generate_greedy(model, tok_clean, MAX_NEW_TOKENS, stop_on_newline=True)
        patched_text = generate_greedy(
            model, tok_clean, MAX_NEW_TOKENS,
            hook_name=cfg.hook_name, source_resid=corrupt_resid, positions=[plan_pos],
            stop_on_newline=True,
        )
        control_text = generate_greedy(
            model, tok_clean, MAX_NEW_TOKENS,
            hook_name=cfg.hook_name, source_resid=corrupt_resid, positions=[early_pos],
            stop_on_newline=True,
        )

        baseline_family = _classify_rhyme(baseline_text, rhyme_a, rhyme_b)
        patched_family = _classify_rhyme(patched_text, rhyme_a, rhyme_b)
        control_family = _classify_rhyme(control_text, rhyme_a, rhyme_b)

        per_item.append(
            {
                "id": item["id"],
                "length_matched": length_matched,
                "plan_position": plan_pos,
                "early_position": early_pos,
                "probe_probability_at_plan": {k: round(v, 5) for k, v in probe_probs.items()},
                "top_sae_features_at_plan": top_features,
                "generated": {
                    "baseline": {"text": baseline_text, "family": baseline_family},
                    "patched_at_plan_position": {"text": patched_text, "family": patched_family},
                    "control_early_position": {"text": control_text, "family": control_family},
                },
                "flipped_to_corrupt": baseline_family == "a" and patched_family == "b",
                "control_flipped": baseline_family == "a" and control_family == "b",
            }
        )

    n = len(per_item)
    flips = sum(1 for it in per_item if it["flipped_to_corrupt"])
    control_flips = sum(1 for it in per_item if it["control_flipped"])
    probe_a = [it["probe_probability_at_plan"].get(items[i]["probe_words"][0], 0.0)
               for i, it in enumerate(per_item)]
    probe_b = [it["probe_probability_at_plan"].get(items[i]["probe_words"][1], 0.0)
               for i, it in enumerate(per_item)]
    aggregate = {
        "n_items": n,
        "mean_probe_prob_family_a": round(sum(probe_a) / n, 5) if n else 0.0,
        "mean_probe_prob_family_b": round(sum(probe_b) / n, 5) if n else 0.0,
        "planning_flip_rate": round(flips / n, 4) if n else 0.0,
        "control_flip_rate": round(control_flips / n, 4) if n else 0.0,
        "planning_effect": round((flips - control_flips) / n, 4) if n else 0.0,
    }

    params = {
        "layer": cfg.layer,
        "hook": cfg.hook_name,
        "sae": {"release": cfg.sae_release, "id": cfg.sae_id, "trained_by_us": False},
        "top_k": top_k,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    run_dir = new_run_dir("exp2_planning")
    results = emit(
        "exp2_planning", run_dir,
        model=cfg.hf_name, params=params, per_item=per_item, aggregate=aggregate,
        summary_md=_summary,
    )
    print(f"[exp2] wrote {run_dir}/results.json and summary.md")
    print(f"[exp2] planning_flip_rate={aggregate['planning_flip_rate']} "
          f"control_flip_rate={aggregate['control_flip_rate']} "
          f"planning_effect={aggregate['planning_effect']}")
    return results


def _summary(r: dict) -> str:
    agg = r["aggregate"]
    lines = [
        "# Experiment 2: planning ahead (rhyming couplets)",
        "",
        f"- Model: `{r['model']}`  |  layer {r['params']['layer']} (`{r['params']['hook']}`)",
        f"- SAE: `{r['params']['sae']['release']}` / `{r['params']['sae']['id']}`",
        f"- Items: {agg['n_items']}",
        "",
        f"**Planning flip rate (patch at planning position): {agg['planning_flip_rate']}**",
        f"**Control flip rate (patch at an early position): {agg['control_flip_rate']}**",
        f"**Planning effect (difference): {agg['planning_effect']}**",
        "",
        "| item | baseline | patched@plan | control@early | flipped |",
        "|---|---|---|---|---|",
    ]
    for it in r["per_item"]:
        g = it["generated"]
        lines.append(
            f"| {it['id']} | {g['baseline']['family']} `{g['baseline']['text'].strip()!r}` "
            f"| {g['patched_at_plan_position']['family']} `{g['patched_at_plan_position']['text'].strip()!r}` "
            f"| {g['control_early_position']['family']} `{g['control_early_position']['text'].strip()!r}` "
            f"| {'yes' if it['flipped_to_corrupt'] else 'no'} |"
        )
    lines += [
        "",
        "## Reading this",
        "",
        "A planning_effect well above 0 means patching the model's activations at the",
        "newline -- before any of the second line has been generated -- with a",
        "different rhyme family's activations reliably changes which family the",
        "generated line ends in, and that a length-matched early-position control",
        "does not do the same. That is causal evidence the line ending is decided",
        "at (or before) the newline, i.e. the model plans ahead rather than",
        "greedily improvising word by word.",
        "",
        "Full per-item probe probabilities and SAE features are in `results.json`.",
    ]
    return "\n".join(lines)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--layer", type=int, default=12, help="residual-stream layer to probe (default 12)")
    p.add_argument("--top-k", type=int, default=32, help="top SAE features to record (default 32)")
