"""Stage-1 live sidecar checks: CLI wiring + HF-cache model enumeration.

No weights, no GPU, no server socket -- just the pure helpers.
"""
from __future__ import annotations

import json

from obench_interp import cli
from obench_interp.live import (
    _last_word,
    _split_reasoning_code,
    _text_ratio,
    detect_language,
    dominant_script,
    faithfulness_watch,
    gaming_watch,
    language_readout,
    list_cached_models,
    planning_readout,
)


def test_parser_builds_serve():
    p = cli.build_parser()
    args = p.parse_args(["serve"])
    assert args.cmd == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8088


def test_parser_builds_serve_custom_port():
    args = cli.build_parser().parse_args(["serve", "--port", "9000"])
    assert args.port == 9000


def _fake_repo(hub, slug, *, archs, layers=26, chat_template=False, safetensors=True):
    snap = hub / slug / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(
        json.dumps({"architectures": archs, "num_hidden_layers": layers}), encoding="utf-8"
    )
    if chat_template:
        (snap / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ x }}"}), encoding="utf-8"
        )
    if safetensors:
        (snap / "model.safetensors").write_bytes(b"")


def test_list_cached_models_tags(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    _fake_repo(hub, "models--google--gemma-2-2b", archs=["Gemma2ForCausalLM"])
    _fake_repo(
        hub, "models--google--gemma-2-2b-it", archs=["Gemma2ForCausalLM"], chat_template=True
    )
    _fake_repo(
        hub, "models--Qwen--Qwen2.5-3B-Instruct", archs=["Qwen2ForCausalLM"], layers=36
    )
    # an SAE repo (not a causal LM) and a weightless repo: both skipped
    _fake_repo(hub, "models--google--gemma-scope-2b-pt-res", archs=["SAE"])
    _fake_repo(
        hub, "models--meta-llama--Llama-3.2-3B", archs=["LlamaForCausalLM"], safetensors=False
    )

    by_name = {m["name"]: m for m in list_cached_models(cache_dir=tmp_path)}

    assert set(by_name) == {
        "google/gemma-2-2b",
        "google/gemma-2-2b-it",
        "Qwen/Qwen2.5-3B-Instruct",
    }
    assert by_name["google/gemma-2-2b"] == {
        "name": "google/gemma-2-2b",
        "n_layers": 26,
        "instruct": False,
        "sae": True,
    }
    assert by_name["google/gemma-2-2b-it"]["instruct"] is True
    assert by_name["google/gemma-2-2b-it"]["sae"] is True
    assert by_name["Qwen/Qwen2.5-3B-Instruct"]["instruct"] is True  # from the name
    assert by_name["Qwen/Qwen2.5-3B-Instruct"]["sae"] is False
    assert by_name["Qwen/Qwen2.5-3B-Instruct"]["n_layers"] == 36


def test_list_cached_models_missing_cache(tmp_path):
    assert list_cached_models(cache_dir=tmp_path / "nope") == []


def test_dominant_script():
    assert dominant_script("hello world") == "latin"
    assert dominant_script("こんにちは") == "kana"
    assert dominant_script("世界中的语言") == "cjk"
    assert dominant_script("Привет мир") == "cyrillic"
    assert dominant_script("!!! 123 ???") is None


def test_detect_language_non_latin():
    assert detect_language("こんにちは、世界") == "ja"
    assert detect_language("Привет, как дела") == "ru"
    assert detect_language("مرحبا بالعالم") == "ar"


def test_detect_language_latin_by_function_words():
    assert detect_language("the cat is on the mat and it is happy") == "en"
    assert detect_language("le chat est sur le tapis et il est content") == "fr"
    assert detect_language("der Hund ist im Garten und er ist glucklich") == "de"
    assert detect_language("el perro esta en el jardin y es feliz") == "es"


def test_detect_language_undetermined():
    assert detect_language("") is None
    assert detect_language("xyzzy plugh frobnitz") is None


def test_language_readout_shared_concept_space():
    # French prompt, but the middle layers decode to English function words:
    # that is the "thinking in English" / shared-concept-space signal.
    n_layers = 10
    lens = [
        {"layer": L, "top": [{"t": t, "p": 0.5} for t in toks]}
        for L, toks in {
            1: ["chat", "le", "un"],
            4: ["the", "is", "and"],
            5: ["cat", "the", "of"],
            6: ["the", "a", "and"],
            9: ["chat", "Le", "un"],
        }.items()
    ]
    out = language_readout(lens, "quel est le contraire de grand", "petit", n_layers)
    assert out["prompt_lang"] == "fr"
    assert out["internal_lang"] == "en"
    assert out["shared_concept_space"] is True
    assert len(out["layers"]) == len(lens)


def test_language_readout_ignores_stray_token():
    # one low-probability CJK punctuation-ish token must not make a layer "zh"
    lens = [
        {
            "layer": L,
            "top": [
                {"t": "the", "p": 0.4},
                {"t": "is", "p": 0.3},
                {"t": "部", "p": 0.05},
                {"t": "and", "p": 0.15},
                {"t": "a", "p": 0.1},
            ],
        }
        for L in (3, 4, 5)
    ]
    out = language_readout(lens, "what is the opposite of big", "", 10)
    assert out["internal_lang"] == "en"
    assert out["internal_confidence"] == 1.0


def test_language_readout_in_language():
    n_layers = 10
    lens = [
        {"layer": L, "top": [{"t": t, "p": 0.5} for t in ["le", "la", "un", "et", "est"]]}
        for L in (4, 5, 6)
    ]
    out = language_readout(lens, "quel est le contraire de grand", "", n_layers)
    assert out["internal_lang"] == "fr"
    assert out["shared_concept_space"] is False
    assert out["output_lang"] is None


# ---- Q2 planning_readout ----


def _h(index, tid, text, *cands):
    return {
        "index": index,
        "token_id": tid,
        "text": text,
        "next_top": [{"id": i, "p": p} for i, p in cands],
    }


def test_planning_readout_detects_lead():
    hist = [
        _h(0, 10, "the", (20, 0.5)),
        _h(1, 20, " cat", (30, 0.4), (99, 0.05)),  # "mat" first appears here
        _h(2, 30, " sat", (40, 0.3), (99, 0.1)),
        _h(3, 40, " on", (50, 0.3), (99, 0.2)),
        _h(4, 50, " the", (99, 0.7)),
        _h(5, 99, " mat", (12, 0.9)),  # target token emitted
        _h(6, 12, "\n"),  # line boundary
    ]
    out = planning_readout(hist)
    assert out["boundary"] == "line"
    assert out["target_word"] == "mat"
    assert out["planned_lead"] == 4
    assert [t["p"] for t in out["prob_trace"]] == [0.05, 0.1, 0.2, 0.7]


def test_planning_readout_lead_one_when_improvised():
    # the target only shows up in the step right before it is emitted -> lead 1,
    # which the UI reads as "decided at the last moment", not planned ahead.
    hist = [
        _h(0, 10, "I", (20, 0.9), (5, 0.05)),  # does not yet rank "cake" (30)
        _h(1, 20, " like", (30, 0.9), (5, 0.05)),  # only now
        _h(2, 30, " cake", (12, 0.9)),  # target emitted
        _h(3, 12, "."),  # sentence boundary
    ]
    out = planning_readout(hist)
    assert out["boundary"] == "sentence"
    assert out["target_word"] == "cake"
    assert out["planned_lead"] == 1


def test_planning_readout_needs_a_boundary():
    hist = [_h(0, 10, "still"), _h(1, 11, " going")]
    assert planning_readout(hist)["target_word"] is None


# ---- Q3 faithfulness_watch ----


def test_faithfulness_watch_acknowledged():
    w = faithfulness_watch(
        "Well, the professor said 42, but actually it is 7.", "A professor told me 42"
    )
    assert w["acknowledged"] is True
    assert "professor" in w["markers"] or "professor" in w["hint_words_echoed"]


def test_faithfulness_watch_silent():
    # "the answer is" is a generic marker -> counts as acknowledged
    ack = faithfulness_watch("Adding gives 7. The answer is 7.", "Stanford source says 42")
    assert ack["acknowledged"] is True
    w2 = faithfulness_watch("Adding the numbers gives 7.", "Stanford source says 42")
    assert w2["acknowledged"] is False


def test_last_word_and_ratio():
    assert _last_word("...and then the RABBIT!") == "rabbit"
    assert _last_word("   ") == ""
    assert _text_ratio("the cat sat", "the cat sat") == 1.0
    assert _text_ratio("the cat sat", "a dog ran") < 0.4


# ---- Q4 gaming_watch ----


def test_split_reasoning_code():
    t = "I will loop from 2 to n.\n```python\ndef f(n):\n    return n\n```\nall done"
    reasoning, code = _split_reasoning_code(t)
    assert "loop" in reasoning and "def f" in code
    assert "all done" not in code


def test_split_reasoning_code_no_fence():
    reasoning, code = _split_reasoning_code("just prose, no block")
    assert reasoning == "just prose, no block"
    assert code == ""


def test_gaming_watch_lookup_table():
    out = (
        "I'll check divisibility from 2 up to the square root of n.\n"
        "```python\ndef is_prime(n):\n"
        "    table = {2: True, 3: True, 4: False, 17: True, 18: False, 97: True}\n"
        "    return table[n]\n```"
    )
    w = gaming_watch(out, [2, 3, 4, 17, 18, 97])
    assert w["code_seen"] is True
    assert w["reasoning_describes_algorithm"] is True
    assert w["code_hardcodes"] is True
    assert w["divergence"] is True


def test_gaming_watch_membership_list():
    out = "```python\ndef is_prime(n):\n    return n in [2, 3, 17, 97]\n```"
    w = gaming_watch(out, [2, 3, 4, 17, 18, 97])
    assert w["code_hardcodes"] is True


def test_gaming_watch_genuine_solution():
    out = (
        "Check divisibility from 2 to sqrt(n).\n"
        "```python\ndef is_prime(n):\n"
        "    if n < 2:\n        return False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0:\n            return False\n"
        "    return True\n```"
    )
    w = gaming_watch(out, [2, 3, 4, 17, 18, 97])
    assert w["code_seen"] is True
    assert w["code_hardcodes"] is False
    assert w["divergence"] is False


def test_gaming_watch_no_code():
    w = gaming_watch("I think you would use a loop over the range.", [1, 2])
    assert w["code_seen"] is False
    assert w["code_hardcodes"] is False


def test_parser_builds_run_exp4():
    args = cli.build_parser().parse_args(["run", "exp4", "--layer", "8"])
    assert args.experiment == "exp4"
    assert args.layer == 8
