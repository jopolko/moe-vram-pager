"""Experiment 1: does the model share concept representations across languages?

Method
------
1. Load gemma-2-2b (TransformerLens), hook the layer-12 residual stream.
2. For each concept (an antonym task) in each language, run the prompt and grab
   the residual activation at the final prompt token, i.e. the position that
   predicts the answer.
3. Encode that activation through the Gemma Scope SAE for this layer.
4. Compare the SAE feature vectors across languages:
     - cosine similarity of the full feature-activation vectors, per language pair
     - Jaccard overlap of the top-k firing feature ids, per language pair
5. Aggregate: which features fire (above threshold) for the SAME concept in
   ALL languages? Those are the candidate language-agnostic concept features.

A high mean cross-language cosine + a non-trivial set of all-language features is
evidence the model represents the concept in a shared, language-independent way.

Output: results.json (stable schema) + summary.md, under results/exp1_multilingual/.
"""
from __future__ import annotations

import argparse
import itertools
import json

from ..config import ModelConfig
from ..env import INTERP_ROOT
from ..loader import load_model, load_sae
from ..report import emit, new_run_dir

DATA = INTERP_ROOT / "data" / "multilingual_antonyms.json"


def _cosine(a, b):
    import torch

    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def run(args: argparse.Namespace) -> dict:
    import torch

    cfg = ModelConfig(layer=args.layer, hook_name=f"blocks.{args.layer}.hook_resid_post")
    prompts = json.loads(DATA.read_text(encoding="utf-8"))
    languages = args.languages or sorted({lang for it in prompts["items"] for lang in it["prompts"]})
    top_k = args.top_k

    print(f"[exp1] loading {cfg.hf_name} ...")
    model = load_model(cfg)
    print(f"[exp1] loading SAE {cfg.sae_release} / {cfg.sae_id} ...")
    sae = load_sae(cfg)

    per_concept: list[dict] = []
    # feature id -> set of (concept, lang) it fired in, for the aggregate pass
    fired: dict[int, set[tuple[str, str]]] = {}

    for item in prompts["items"]:
        concept = item["concept"]
        feats_by_lang: dict[str, "torch.Tensor"] = {}
        top_by_lang: dict[str, list[dict]] = {}
        predicted: dict[str, str] = {}

        for lang in languages:
            text = item["prompts"].get(lang)
            if text is None:
                continue
            tokens = model.to_tokens(text)
            with torch.no_grad():
                logits, cache = model.run_with_cache(
                    tokens, names_filter=cfg.hook_name, return_type="logits"
                )
            resid = cache[cfg.hook_name][0, -1]  # (d_model,) at the answer position
            with torch.no_grad():
                acts = sae.encode(resid.to(sae.W_enc.dtype)).float()  # (d_sae,)

            feats_by_lang[lang] = acts
            next_id = int(logits[0, -1].argmax())
            predicted[lang] = model.to_string(torch.tensor([next_id]))

            topv, topi = acts.topk(top_k)
            top_by_lang[lang] = [
                {"id": int(i), "act": round(float(v), 4)} for v, i in zip(topv, topi) if v > 0
            ]
            for entry in top_by_lang[lang]:
                fired.setdefault(entry["id"], set()).add((concept, lang))

        pair_cos: dict[str, float] = {}
        pair_shared: dict[str, list[int]] = {}
        for l1, l2 in itertools.combinations(feats_by_lang, 2):
            key = f"{l1}-{l2}"
            pair_cos[key] = round(_cosine(feats_by_lang[l1], feats_by_lang[l2]), 4)
            s1 = {e["id"] for e in top_by_lang[l1]}
            s2 = {e["id"] for e in top_by_lang[l2]}
            pair_shared[key] = sorted(s1 & s2)

        per_concept.append(
            {
                "concept": concept,
                "predicted_token": predicted,
                "language_pair_cosine": pair_cos,
                "language_pair_jaccard": {
                    k: round(_jaccard({e["id"] for e in top_by_lang[k.split("-")[0]]},
                                      {e["id"] for e in top_by_lang[k.split("-")[1]]}), 4)
                    for k in pair_cos
                },
                "shared_top_features": pair_shared,
                "top_features_by_lang": top_by_lang,
            }
        )

    langs_present = {c["concept"]: set(c["top_features_by_lang"]) for c in per_concept}
    all_lang_features: dict[str, list[int]] = {}
    for concept, present in langs_present.items():
        feats = [
            fid
            for fid, hits in fired.items()
            if {lang for (cc, lang) in hits if cc == concept} >= present
        ]
        all_lang_features[concept] = sorted(feats)

    cos_values = [v for c in per_concept for v in c["language_pair_cosine"].values()]
    aggregate = {
        "mean_cross_language_cosine": round(sum(cos_values) / len(cos_values), 4) if cos_values else 0.0,
        "language_agnostic_features_by_concept": all_lang_features,
        "language_agnostic_feature_count": sum(len(v) for v in all_lang_features.values()),
    }

    params = {
        "layer": cfg.layer,
        "hook": cfg.hook_name,
        "sae": {"release": cfg.sae_release, "id": cfg.sae_id, "trained_by_us": False},
        "languages": languages,
        "top_k": top_k,
        "concepts": [c["concept"] for c in per_concept],
    }

    run_dir = new_run_dir("exp1_multilingual")
    results = emit(
        "exp1_multilingual",
        run_dir,
        model=cfg.hf_name,
        params=params,
        per_item=per_concept,
        aggregate=aggregate,
        summary_md=_summary,
    )
    print(f"[exp1] wrote {run_dir}/results.json and summary.md")
    print(f"[exp1] mean cross-language cosine = {aggregate['mean_cross_language_cosine']}, "
          f"{aggregate['language_agnostic_feature_count']} all-language features")
    return results


def _summary(r: dict) -> str:
    p = r["params"]
    lines = [
        "# Experiment 1: multilingual concept sharing",
        "",
        f"- Model: `{r['model']}`  |  layer {p['layer']} (`{p['hook']}`)",
        f"- SAE: `{p['sae']['release']}` / `{p['sae']['id']}`",
        f"- Languages: {', '.join(p['languages'])}  |  top-k features: {p['top_k']}",
        "",
        f"**Mean cross-language cosine similarity: {r['aggregate']['mean_cross_language_cosine']}**",
        f"**All-language concept features: {r['aggregate']['language_agnostic_feature_count']}**",
        "",
        "| concept | mean pair cosine | all-language features | predicted tokens |",
        "|---|---|---|---|",
    ]
    for c in r["per_item"]:
        cos = c["language_pair_cosine"].values()
        mean_cos = round(sum(cos) / len(cos), 3) if cos else 0.0
        alf = r["aggregate"]["language_agnostic_features_by_concept"].get(c["concept"], [])
        toks = " / ".join(f"{k}:{v.strip()!r}" for k, v in c["predicted_token"].items())
        lines.append(f"| {c['concept']} | {mean_cos} | {len(alf)} | {toks} |")
    lines += [
        "",
        "## Reading this",
        "",
        "High mean cosine (say > 0.5) plus a non-trivial all-language feature set",
        "is evidence the concept lives in a shared, language-independent representation.",
        "Low cosine with correct predicted tokens in every language suggests the model",
        "solves the task per-language without a common internal concept.",
        "",
        "Full per-pair numbers and firing features are in `results.json`.",
    ]
    return "\n".join(lines)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--layer", type=int, default=12, help="residual-stream layer to probe (default 12)")
    p.add_argument("--top-k", type=int, default=32, help="top firing features to compare (default 32)")
    p.add_argument(
        "--languages", nargs="*", default=None, help="subset of languages (default: all in the data file)"
    )
