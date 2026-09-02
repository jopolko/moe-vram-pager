"""Experiment implementations. Each exposes `run(args) -> dict` and `add_args(p)`,
and writes a timestamped result dir under interpretability/results/ through
`report.emit()` (see docs/result-schema.md for the shared envelope).

  exp1_multilingual      cross-language concept sharing (Gemma Scope SAE)
  exp2_planning           planning ahead in generation (activation patching)
  exp3_cot_faithfulness   chain-of-thought faithfulness (activation patching)
"""
