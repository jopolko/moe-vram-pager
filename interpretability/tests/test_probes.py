"""Probe framework checks that need no model or GPU: fit on synthetic
activations, save/load roundtrip, ProbeSet scoring, dataset shape.
"""
from __future__ import annotations

import json

import numpy as np

from obench_interp import probes as P
from obench_interp.env import INTERP_ROOT


def _blobs(n_per_class, n_layers, hidden, sep, seed=0):
    """Two Gaussian blobs, separable from layer `n_layers // 2` on."""
    rng = np.random.default_rng(seed)
    acts = rng.standard_normal((2 * n_per_class, n_layers, hidden)).astype(np.float32)
    labels = ["a"] * n_per_class + ["b"] * n_per_class
    for li in range(n_layers // 2, n_layers):
        acts[n_per_class:, li, 0] += sep  # push class "b" apart on dim 0
    return acts, labels, list(range(n_layers))


def test_fit_picks_a_separable_layer():
    acts, labels, layers = _blobs(40, 6, 16, sep=8.0)
    probe = P._fit("t", "m/x", "binary", acts, labels, layers, dataset_sha="abc")
    assert probe.classes == ["a", "b"]
    assert probe.layer >= 3  # the separable half
    assert probe.trained_layers == layers
    assert probe.cv_accuracy > 0.9
    assert probe.w.shape == (6, 2, 16)
    assert len(probe.layer_accuracy) == 6


def test_fit_band_restricts_the_headline_layer():
    # make an EARLY layer the most separable, but restrict the band to the tail
    acts, labels, layers = _blobs(30, 8, 12, sep=0.0)
    acts[15:, 0, 0] += 12.0  # layer 0 perfectly separable
    acts[15:, 6, 0] += 3.0
    probe = P._fit("t", "m/x", "binary", acts, labels, layers, dataset_sha="d", band=(0.6, 1.0))
    assert probe.layer >= 4  # band kept it out of the trivial early layer


def test_save_load_roundtrip(tmp_path):
    acts, labels, layers = _blobs(30, 4, 12, sep=6.0)
    probe = P._fit("lang", "m/x", "multiclass", acts, labels, layers, dataset_sha="d")
    P.save(probe, tmp_path)
    assert (tmp_path / "lang.npz").is_file() and (tmp_path / "lang.json").is_file()

    (tmp_path / "google__gemma-2-2b-it").mkdir()  # load_set keys by slug
    P.save(probe, tmp_path / "google__gemma-2-2b-it")
    orig_dir = P.PROBES_DIR
    try:
        P.PROBES_DIR = tmp_path
        pset = P.load_set("google/gemma-2-2b-it")
    finally:
        P.PROBES_DIR = orig_dir
    assert "lang" in pset
    got = pset.probes["lang"]
    assert got.layer == probe.layer and got.trained_layers == probe.trained_layers
    assert np.allclose(got.w, probe.w, atol=1e-2)  # fp16 on disk, fp32 in memory


def test_probeset_score_shapes():
    acts, labels, layers = _blobs(30, 5, 10, sep=7.0)
    probe = P._fit("sycophancy", "m/x", "binary", acts, labels, layers, dataset_sha="d")
    probe.classes[:] = ["caved", "firm"]
    pset = P.ProbeSet({"sycophancy": probe})
    assert pset.layers == {probe.layer}

    resid = {L: np.zeros(10, np.float32) for L in range(5)}
    out = pset.score("sycophancy", resid)
    assert set(out["p"]) == {"caved", "firm"}
    assert len(out["per_layer"]) == 5
    # a layer the probe was never trained on is skipped, not crashed
    assert len(pset.score("sycophancy", {99: np.zeros(10, np.float32)})["per_layer"]) == 0
    assert pset.score("missing", resid) is None


def test_pick_answer_is_in_probes():
    assert P.pick_answer("it is prime", "composite", "prime") == "prime"


def test_language_dataset_wellformed():
    raw = json.loads((INTERP_ROOT / "data" / "probe_language.json").read_text(encoding="utf-8"))
    for lang, samples in raw["samples"].items():
        assert 2 <= len(lang) <= 3
        assert len(samples) >= 10, f"{lang}: only {len(samples)} samples"
        assert all(isinstance(s, str) and s.strip() for s in samples)


def test_sycophancy_dataset_wellformed():
    raw = json.loads((INTERP_ROOT / "data" / "probe_sycophancy.json").read_text(encoding="utf-8"))
    styles = set(raw["pressure_templates"])
    assert all("{ans}" in t for t in raw["pressure_templates"].values())
    seen = set()
    for it in raw["items"]:
        assert it["id"] not in seen
        seen.add(it["id"])
        assert it["correct_answer"].lower() != it["pushed_answer"].lower()
        assert set(it["styles"]) <= styles
