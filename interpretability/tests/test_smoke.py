"""Cheap checks that don't need weights or a GPU."""
from __future__ import annotations

import json

from obench_interp import cli, config
from obench_interp.env import INTERP_ROOT, run_all
from obench_interp.experiments.exp2_planning import _classify_rhyme
from obench_interp.report import envelope, write


def test_parser_builds_exp1():
    p = cli.build_parser()
    args = p.parse_args(["run", "exp1", "--layer", "10"])
    assert args.layer == 10
    assert args.experiment == "exp1"


def test_parser_builds_exp2():
    p = cli.build_parser()
    args = p.parse_args(["run", "exp2", "--layer", "8", "--top-k", "16", "--model", "Qwen/Qwen2.5-0.5B-Instruct"])
    assert args.layer == 8
    assert args.top_k == 16
    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args.experiment == "exp2"


def test_parser_builds_exp3():
    p = cli.build_parser()
    args = p.parse_args(["run", "exp3"])
    assert args.layer is None  # resolved to the model's middle layer at run time
    assert args.model == config.EXP3_DEFAULT_MODEL
    assert args.experiment == "exp3"


def test_config_hook_matches_layer():
    c = config.ModelConfig()
    assert c.hook_name == f"blocks.{c.layer}.hook_resid_post"
    assert config.INSTRUCT.hook_name == f"blocks.{config.INSTRUCT.layer}.hook_resid_post"


def test_prompt_data_parallel():
    data = json.loads((INTERP_ROOT / "data" / "multilingual_antonyms.json").read_text(encoding="utf-8"))
    langs = {lang for it in data["items"] for lang in it["prompts"]}
    for it in data["items"]:
        assert set(it["prompts"]) == langs, f"{it['concept']} missing a language"


def test_rhyming_couplets_data():
    data = json.loads((INTERP_ROOT / "data" / "rhyming_couplets.json").read_text(encoding="utf-8"))
    assert data["items"], "no couplet items"
    for it in data["items"]:
        assert it["line1_clean"] and it["line1_corrupt"]
        assert it["rhyme_a"] and it["rhyme_b"]
        assert len(it["probe_words"]) == 2
        # probe words should belong to the two families they're meant to track
        assert it["probe_words"][0].lower() in [w.lower() for w in it["rhyme_a"]]
        assert it["probe_words"][1].lower() in [w.lower() for w in it["rhyme_b"]]


def test_cot_hinted_questions_data():
    data = json.loads((INTERP_ROOT / "data" / "cot_hinted_questions.json").read_text(encoding="utf-8"))
    templates = data["hint_templates"]
    assert data["items"], "no question items"
    for it in data["items"]:
        assert it["hint_style"] in templates, f"{it['id']}: unknown hint_style {it['hint_style']}"
        assert it["hint_answer"].strip().lower() != it["correct"].strip().lower(), (
            f"{it['id']}: hint_answer equals correct, defeats the point of the item"
        )
        assert "{ans}" in templates[it["hint_style"]]


def test_classify_rhyme():
    assert _classify_rhyme(" rabbit.", ["rabbit", "habit"], ["ground", "found"]) == "a"
    assert _classify_rhyme(" the ground.", ["rabbit", "habit"], ["ground", "found"]) == "b"
    assert _classify_rhyme(" he had to grab it.", ["grab it", "nab it"], ["found it"]) == "a"
    assert _classify_rhyme(" a purple elephant.", ["rabbit"], ["ground"]) == "other"


def test_report_envelope_schema(tmp_path):
    results = envelope(
        "exp_test", model="dummy/model", params={"layer": 1}, per_item=[{"a": 1}], aggregate={"n": 1}
    )
    for key in ("schema_version", "experiment", "generated_at", "model", "params", "per_item", "aggregate"):
        assert key in results
    write(tmp_path, results, "# summary")
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == "# summary"


def test_doctor_runs():
    checks = run_all()
    assert any(c.name == "deps" for c in checks)
