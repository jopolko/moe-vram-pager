# Experiment 2: planning ahead (rhyming couplets)

Run: `./interp run exp2 [--layer 12] [--top-k 32]`

## Plan (from the build spec)

1. Load gemma-2-2b (TransformerLens), hook the layer-12 residual stream.
2. For each item in `../../data/rhyming_couplets.json`, feed
   `line1_clean + "\n"` and its length-matched `line1_corrupt + "\n"`.
3. Cache the residual stream for both at the "planning position" (the last
   prompt token, i.e. the newline slot -- none of line 2 exists yet).
4. **Observational**: read the model's own next-token probability for each
   rhyme family's `probe_words` at the planning position.
5. **SAE**: encode the clean planning-position activation through the Gemma
   Scope SAE; record top firing feature ids.
6. **Causal**: generate line 2 from the clean prompt three ways -- baseline,
   patched (splice the corrupt run's residual stream in at the planning
   position), control (same splice, but at an early line-1 position). Classify
   each generated line's ending against `rhyme_a` / `rhyme_b`.
7. Emit `results/exp2_planning/<timestamp>/{results.json,summary.md}`.

## Reading the output

See `../../docs/result-schema.md` for the shared top-level envelope. Per-item
fields specific to exp2:

```json
{
  "id": "rabbit_carrot",
  "plan_position": 11,
  "probe_probability_at_plan": {"rabbit": 0.31, "ground": 0.02},
  "top_sae_features_at_plan": [{"id": 4821, "act": 3.1}],
  "generated": {
    "baseline": {"text": " rabbit.", "family": "a"},
    "patched_at_plan_position": {"text": " ground.", "family": "b"},
    "control_early_position": {"text": " rabbit.", "family": "a"}
  },
  "flipped_to_corrupt": true,
  "control_flipped": false
}
```

`aggregate.planning_effect` is the headline number: `planning_flip_rate -
control_flip_rate`. Well above 0 is causal evidence the model decides the
line ending at (or before) the newline rather than improvising word by word.
Near 0 means the observational probe numbers, if elevated, are more likely
surface priming than a committed plan.

## Caveats

- `rhyme_a`/`rhyme_b` in `data/rhyming_couplets.json` are hand-written
  candidate word/phrase lists; `_classify_rhyme` does a normalized-suffix
  match, not a phonetic rhyme check. Expand the lists if a model's actual
  wording (a synonym, a different phrase) isn't being classified.
- `line1_clean` / `line1_corrupt` are matched by word count, not guaranteed
  equal token count after tokenization. `length_matched` in each per-item
  result flags mismatches; the patch still runs (using the shorter prompt's
  last-token index) but treat mismatched items as weaker evidence.
