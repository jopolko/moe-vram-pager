# Experiment 1: multilingual concept sharing

Run: `./interp run exp1 [--layer 12] [--top-k 32] [--languages en fr de es]`

Model confirmed (see `../../docs/model-choice.md`): `google/gemma-2-2b`.

## Plan (from the build spec)

1. Load model via HF transformers in fp16, wrap with TransformerLens, hook a
   middle-layer residual stream (`blocks.{L}.hook_resid_post`).
2. Feed the semantically-parallel prompts in `../../data/multilingual_antonyms.json`
   (4-5 languages, same antonym task).
3. Capture the residual-stream activation at the answer-predicting token for each
   language variant.
4. Load a pretrained SAE for this layer if one exists publicly (Gemma Scope for
   gemma-2-2b); otherwise train a small SAE on this layer with `sae_lens`. Check
   availability first.
5. Encode each activation through the SAE. Compute cross-language feature overlap:
   - cosine similarity of SAE feature-activation vectors between language pairs
   - shared top-k active features (Jaccard over the top-k firing feature ids)
6. Emit:
   - `results/exp1_multilingual/<timestamp>/results.json` (structured, see schema below)
   - `results/exp1_multilingual/<timestamp>/summary.md` (human-readable)

## Output schema

See `../../docs/result-schema.md` for the shared top-level envelope
(`schema_version`, `experiment`, `model`, `params`, `per_item`, `aggregate`).
exp1's `params` carries `layer`, `hook`, `sae`, `languages`, `top_k`,
`concepts`. Each `per_item` entry (one per concept) looks like:

```json
{
  "concept": "small->big",
  "predicted_token": {"en": "big", "fr": "grand", "...": "..."},
  "language_pair_cosine": {"en-fr": 0.81, "en-de": 0.77, "...": 0.0},
  "language_pair_jaccard": {"en-fr": 0.4, "...": 0.0},
  "shared_top_features": {"en-fr": [1234, 5678], "...": []},
  "top_features_by_lang": {"en": [ {"id": 1234, "act": 4.2} ]}
}
```

`aggregate` adds `mean_cross_language_cosine`,
`language_agnostic_features_by_concept`, `language_agnostic_feature_count`.
