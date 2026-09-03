"""Live sidecar checks that don't need weights or a GPU: CLI wiring + HF-cache
model enumeration. The per-token signals are trained probes now (see
test_probes.py); there are no text-heuristic helpers left to unit test.
"""
from __future__ import annotations

import json

from obench_interp import cli
from obench_interp.live import list_cached_models
from obench_interp.probes import pick_answer


def test_parser_builds_serve():
    p = cli.build_parser()
    args = p.parse_args(["serve"])
    assert args.cmd == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8088


def test_parser_builds_serve_custom_port():
    args = cli.build_parser().parse_args(["serve", "--port", "9000"])
    assert args.port == 9000


def test_parser_builds_train_probes():
    args = cli.build_parser().parse_args(
        ["train-probes", "--model", "google/gemma-2-2b-it", "--probe", "language"]
    )
    assert args.cmd == "train-probes"
    assert args.model == "google/gemma-2-2b-it"
    assert args.probe == "language"
    assert args.layer_stride == 1


def test_pick_answer():
    assert pick_answer("After checking, 27 is composite.", "composite", "prime") == "composite"
    assert pick_answer("You're right, it is prime.", "composite", "prime") == "prime"
    assert pick_answer("The capital of Australia is Canberra.", "Canberra", "Sydney") == "Canberra"
    assert pick_answer("I am not sure.", "yes", "no") == ""  # "no" must not hit "not"


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
    _fake_repo(hub, "models--google--gemma-scope-2b-pt-res", archs=["SAE"])
    _fake_repo(
        hub, "models--meta-llama--Llama-3.2-3B", archs=["LlamaForCausalLM"], safetensors=False
    )

    by_name = {
        m["name"]: m for m in list_cached_models(cache_dir=tmp_path, probes_dir=tmp_path / "probes")
    }

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
        "probes": [],
    }
    assert by_name["google/gemma-2-2b-it"]["instruct"] is True
    assert by_name["google/gemma-2-2b-it"]["sae"] is True
    assert by_name["Qwen/Qwen2.5-3B-Instruct"]["instruct"] is True
    assert by_name["Qwen/Qwen2.5-3B-Instruct"]["sae"] is False
    assert by_name["Qwen/Qwen2.5-3B-Instruct"]["n_layers"] == 36


def test_list_cached_models_reports_trained_probes(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    _fake_repo(hub, "models--google--gemma-2-2b-it", archs=["Gemma2ForCausalLM"], chat_template=True)
    pdir = tmp_path / "probes" / "google__gemma-2-2b-it"
    pdir.mkdir(parents=True)
    (pdir / "language.json").write_text("{}", encoding="utf-8")

    by_name = {
        m["name"]: m for m in list_cached_models(cache_dir=tmp_path, probes_dir=tmp_path / "probes")
    }
    assert by_name["google/gemma-2-2b-it"]["probes"] == ["language"]


def test_list_cached_models_missing_cache(tmp_path):
    assert list_cached_models(cache_dir=tmp_path / "nope") == []
