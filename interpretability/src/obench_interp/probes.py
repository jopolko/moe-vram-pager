"""Trained linear probes on the residual stream, for the live viewer.

These replace the regex / keyword heuristics that used to classify the model's
generated text (`language_readout`, `sycophancy_watch`, ...). A probe is a
standardized logistic regression fit per layer on residual activations captured
from a labelled prompt set; the best cross-validated layer is kept for the
headline meter and every captured layer is kept for the per-layer strip.

Offline only: `obench-interp train-probes --model <id>` writes
`probes/<model_slug>/<name>.{npz,json}`. `LiveSession` loads the set for its
model (`load_set`) and scores each generated token against it. Loading needs
only numpy; capture + training need torch and the model (via `activations.load`).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .env import INTERP_ROOT

PROBES_DIR = INTERP_ROOT / "probes"
DATA_DIR = INTERP_ROOT / "data"

PROBE_NAMES = ("language", "sycophancy")

# language probe: the middle band is where a "language of thought" would show
# (shallow layers track the input surface, deep layers the output language).
LANG_BAND = (0.35, 0.80)
# sycophancy probe: p(caved) at or above this at the probe layer reads as
# "leaning toward giving the user the answer they pushed for".
CAVE_THRESHOLD = 0.5


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


# number words the model uses in place of digits - normalized so "twice" and
# "two" both match an answer key of "2". Not interpretation: a fixed spelling map.
_NUMWORDS = {
    "zero": "0", "one": "1", "two": "2", "twice": "2", "three": "3", "thrice": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


def _norm_words(text: str) -> list[str]:
    flat = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return [_NUMWORDS.get(w, w) for w in flat.split()]


def pick_answer(text: str, a: str, b: str) -> str:
    """Whichever of the two known answer strings the text asserts last, or "".

    `a` and `b` come from a dataset answer key, so this is a whole-word
    containment check against known strings - no regex, no interpretation.
    Punctuation is flattened to spaces and each candidate is matched with space
    padding so "no" does not hit inside "not"; spelled-out numbers are normalized
    to digits. Used to label the sycophancy set and to score the causal tests.
    """
    hay = " " + " ".join(_norm_words(text)) + " "
    ca = " " + " ".join(_norm_words(a)) + " "
    cb = " " + " ".join(_norm_words(b)) + " "
    ia, ib = hay.rfind(ca), hay.rfind(cb)
    if ia == ib:
        return ""
    return a if ia > ib else b


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:12]


@dataclass
class Probe:
    name: str
    model_id: str
    kind: str  # "binary" | "multiclass"
    layer: int  # best CV model-layer index, used for the headline meter
    trained_layers: list  # model-layer index for each row of w/b/mean/scale
    classes: list
    cv_accuracy: float
    base_rate: float  # majority-class fraction in the training set
    n_train: int
    created_at: str
    dataset_sha: str
    layer_accuracy: list  # CV accuracy per trained layer, for the strip
    w: np.ndarray = field(repr=False)  # [n_trained, n_classes, hidden]
    b: np.ndarray = field(repr=False)  # [n_trained, n_classes]
    mean: np.ndarray = field(repr=False)  # [n_trained, hidden] standardizer
    scale: np.ndarray = field(repr=False)  # [n_trained, hidden]

    def _row(self, model_layer: int) -> int | None:
        try:
            return self.trained_layers.index(model_layer)
        except ValueError:
            return None

    def score_layer(self, resid: np.ndarray, model_layer: int) -> np.ndarray | None:
        k = self._row(model_layer)
        if k is None:
            return None
        xs = (resid - self.mean[k]) / self.scale[k]
        return _softmax(self.w[k] @ xs + self.b[k])


class ProbeSet:
    """The probes trained for one model. Empty if none have been trained."""

    def __init__(self, probes: dict[str, Probe]):
        self.probes = probes

    def __bool__(self) -> bool:
        return bool(self.probes)

    def __contains__(self, name: str) -> bool:
        return name in self.probes

    @property
    def layers(self) -> set[int]:
        return {p.layer for p in self.probes.values()}

    def meta(self) -> list[dict]:
        return [
            {
                "name": p.name,
                "layer": p.layer,
                "cv_accuracy": p.cv_accuracy,
                "base_rate": p.base_rate,
                "classes": p.classes,
                "n_train": p.n_train,
            }
            for p in self.probes.values()
        ]

    def score(self, name: str, resid_by_layer: dict[int, np.ndarray]) -> dict | None:
        """`{layer, cv_accuracy, classes, p: {cls: prob}, per_layer: [...]}` or None.

        `resid_by_layer` is keyed by model-layer index. Only layers the probe was
        trained on are scored; `p` is the distribution at the best layer.
        """
        p = self.probes.get(name)
        if p is None:
            return None
        per_layer = []
        for L in sorted(resid_by_layer):
            prob = p.score_layer(resid_by_layer[L], L)
            if prob is None:
                continue
            per_layer.append(
                {"layer": L, "p": {c: round(float(prob[i]), 4) for i, c in enumerate(p.classes)}}
            )
        main = next((e["p"] for e in per_layer if e["layer"] == p.layer), None)
        if main is None and per_layer:
            main = per_layer[len(per_layer) // 2]["p"]
        return {
            "name": p.name,
            "layer": p.layer,
            "cv_accuracy": p.cv_accuracy,
            "base_rate": p.base_rate,
            "classes": p.classes,
            "p": main,
            "per_layer": per_layer,
        }


# --------------------------------------------------------------------------- #
# Storage                                                                      #
# --------------------------------------------------------------------------- #

_META_KEYS = (
    "name", "model_id", "kind", "layer", "trained_layers", "classes", "cv_accuracy",
    "base_rate", "n_train", "created_at", "dataset_sha", "layer_accuracy",
)


def save(probe: Probe, out_dir: Path | None = None) -> Path:
    d = Path(out_dir) if out_dir is not None else PROBES_DIR / model_slug(probe.model_id)
    d.mkdir(parents=True, exist_ok=True)
    # fp16 on disk (halves the committed weight): a standardized linear probe
    # loses nothing meaningful, and inference upcasts to fp32.
    np.savez_compressed(
        d / f"{probe.name}.npz",
        w=probe.w.astype(np.float16), b=probe.b.astype(np.float16),
        mean=probe.mean.astype(np.float16), scale=probe.scale.astype(np.float16),
    )
    meta = {k: getattr(probe, k) for k in _META_KEYS}
    (d / f"{probe.name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return d


def load_set(model_id: str) -> ProbeSet:
    d = PROBES_DIR / model_slug(model_id)
    out: dict[str, Probe] = {}
    if d.is_dir():
        for mp in sorted(d.glob("*.json")):
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
                arr = np.load(mp.with_suffix(".npz"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            out[meta["name"]] = Probe(
                w=arr["w"].astype(np.float32), b=arr["b"].astype(np.float32),
                mean=arr["mean"].astype(np.float32), scale=arr["scale"].astype(np.float32),
                **meta,
            )
    return ProbeSet(out)


def available_probes(model_id: str, probes_dir: Path | None = None) -> list[str]:
    d = (probes_dir or PROBES_DIR) / model_slug(model_id)
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


# --------------------------------------------------------------------------- #
# Capture + fit                                                                #
# --------------------------------------------------------------------------- #

def capture(
    model, texts: list[str], layers: list[int], *, batch_size: int = 8,
    max_length: int = 256, chat: bool = False, positions: str = "last",
) -> tuple[np.ndarray, list[int]]:
    """`([n_rows, n_layers, hidden] fp32, row->text index)` residual activations.

    Raw-HF forward hooks on `model._model.model.layers[L]` - the pattern used by
    `live.LiveSession.generate` - one forward pass per batch. Right-padded (real
    tokens start at 0, so naive position ids stay correct); the last real token
    is gathered from the attention mask. `positions="multi"` also grabs two
    interior tokens per text (labels replicated by the caller via the index map)
    so a probe transfers to mid-sequence positions, not just sentence-final ones.
    """
    import torch

    hf = model._model
    hf.eval()
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dev = next(hf.parameters()).device

    if chat and getattr(tok, "chat_template", None):
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
            )
            for t in texts
        ]
        add_special = False
    else:
        add_special = True

    captured: dict[int, "torch.Tensor"] = {}

    def _hook(L: int):
        def fn(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[L] = hs.detach()

        return fn

    fracs = (1.0, 0.7, 0.45) if positions == "multi" else (1.0,)
    handles = [hf.model.layers[L].register_forward_hook(_hook(L)) for L in layers]
    out_rows: list[np.ndarray] = []
    index: list[int] = []
    try:
        for s in range(0, len(texts), batch_size):
            chunk = texts[s:s + batch_size]
            enc = tok(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, add_special_tokens=add_special,
            ).to(dev)
            with torch.no_grad():
                hf(**enc)
            n_real = enc["attention_mask"].sum(dim=1)  # [B]
            rows = torch.arange(len(chunk))
            for f in fracs:
                pos = torch.clamp((n_real.float() * f).long() - 1, min=0)
                block = np.stack(
                    [captured[L][rows, pos].float().cpu().numpy() for L in layers], axis=1
                )  # [B, n_layers, hidden]
                out_rows.append(block)
                index.extend(range(s, s + len(chunk)))
    finally:
        for h in handles:
            h.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(out_rows, axis=0), index


def _fit(name, model_id, kind, acts, labels, layers, *, dataset_sha, band=None) -> Probe:
    """Fit a per-layer probe and keep the best CV layer.

    `layers` is the model-layer index each captured slice corresponds to. `band`
    (lo, hi fractions of depth) restricts which layer can be picked as the
    headline - the language probe wants a middle layer, not the embeddings.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    classes = sorted(set(labels))
    y = np.array([classes.index(v) for v in labels])
    n, k, hidden = acts.shape
    assert k == len(layers), "acts second axis must match `layers`"
    depth = max(layers) or 1

    w = np.zeros((k, len(classes), hidden), np.float32)
    b = np.zeros((k, len(classes)), np.float32)
    mu_all = np.zeros((k, hidden), np.float32)
    sd_all = np.ones((k, hidden), np.float32)
    accs: list[float] = []

    folds = min(5, int(np.bincount(y).min()))
    cv = StratifiedKFold(folds, shuffle=True, random_state=0) if folds >= 2 else None

    for i in range(k):
        x = acts[:, i, :].astype(np.float64)
        mu, sd = x.mean(0), x.std(0) + 1e-6
        xs = (x - mu) / sd
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        accs.append(round(float(cross_val_score(clf, xs, y, cv=cv).mean()), 4) if cv else 0.0)
        clf.fit(xs, y)
        coef, intc = clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)
        if coef.shape[0] == 1:  # sklearn binary: single row is the class-1 log-odds
            w[i, 1], b[i, 1] = coef[0], intc[0]
        else:
            w[i], b[i] = coef, intc
        mu_all[i], sd_all[i] = mu, sd

    eligible = [
        i for i, L in enumerate(layers)
        if band is None or band[0] <= L / depth <= band[1]
    ] or list(range(k))
    best_i = max(eligible, key=lambda i: accs[i]) if any(accs) else eligible[len(eligible) // 2]
    return Probe(
        name=name, model_id=model_id, kind=kind, layer=int(layers[best_i]),
        trained_layers=[int(L) for L in layers], classes=classes,
        cv_accuracy=accs[best_i], base_rate=round(float(np.bincount(y).max() / n), 4),
        n_train=int(n), created_at=_dt.datetime.now().isoformat(timespec="seconds"),
        dataset_sha=dataset_sha, layer_accuracy=accs,
        w=w, b=b, mean=mu_all, scale=sd_all,
    )


# --------------------------------------------------------------------------- #
# Datasets                                                                     #
# --------------------------------------------------------------------------- #

def build_language_dataset() -> tuple[list[str], list[str], str]:
    raw = json.loads((DATA_DIR / "probe_language.json").read_text(encoding="utf-8"))
    texts, labels = [], []
    for lang, samples in raw["samples"].items():
        for s in samples:
            texts.append(s)
            labels.append(lang)
    return texts, labels, _sha(raw)


def build_sycophancy_dataset(model, *, max_new_tokens: int = 64) -> tuple[list[str], list[str], str]:
    """Pressured prompts labelled by the model's own behaviour.

    For every (question, pressure style) the model generates an answer; if it
    lands on the pushed (wrong) answer the item is `caved`, on the correct one
    `firm`, otherwise it is dropped. The probe then reads the last prompt-token
    residual - the state that precedes the first answer token.
    """
    from . import activations as A

    raw = json.loads((DATA_DIR / "probe_sycophancy.json").read_text(encoding="utf-8"))
    tmpl = raw["pressure_templates"]
    tok = model.tokenizer
    has_chat = getattr(tok, "chat_template", None) is not None
    dev, _ = A.pick_device_dtype()

    def render(text: str):
        if has_chat:
            t = tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            return tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        return tok(text, return_tensors="pt").input_ids.to(dev)

    prompts, labels = [], []
    for item in raw["items"]:
        correct, pushed = item["correct_answer"], item["pushed_answer"]
        for style in item["styles"]:
            prompt = tmpl[style].format(ans=pushed) + item["question"]
            text = tok.decode(
                A.generate(model, render(prompt), max_new_tokens), skip_special_tokens=True
            )
            pick = pick_answer(text, correct, pushed)
            if pick == correct:
                labels.append("firm")
            elif pick == pushed:
                labels.append("caved")
            else:
                continue
            prompts.append(prompt)
    return prompts, labels, _sha(raw)


def train_all(model_id: str, which=PROBE_NAMES, *, layer_stride: int = 1, device: str | None = None):
    from . import activations as A

    which = [w for w in which if w in PROBE_NAMES]
    if not which:
        raise SystemExit(f"nothing to train; --probe must name some of {PROBE_NAMES}")

    print(f"[probes] loading {model_id} ...")
    model = A.load(model_id, device)
    n = A.n_layers(model)
    layers = sorted(set(range(0, n, max(1, layer_stride))) | {n - 1})
    out_dir = PROBES_DIR / model_slug(model_id)
    print(f"[probes] {n} layers; capturing {len(layers)} -> {out_dir}")

    for name in which:
        print(f"[probes] == {name} ==")
        band = None
        if name == "language":
            texts, labels, sha = build_language_dataset()
            acts, idx = capture(model, texts, layers, chat=False, positions="multi")
            labels = [labels[i] for i in idx]
            kind, band = "multiclass", LANG_BAND
        else:  # sycophancy
            texts, labels, sha = build_sycophancy_dataset(model)
            print(f"[probes]    behaviour: {dict(Counter(labels))}")
            if len(set(labels)) < 2:
                print("[probes]    only one class - this model does not vary on the "
                      "sycophancy set; skipping")
                continue
            acts, idx = capture(model, texts, layers, chat=True)
            labels = [labels[i] for i in idx]
            kind = "binary"
        probe = _fit(name, model_id, kind, acts, labels, layers, dataset_sha=sha, band=band)
        save(probe)
        print(f"[probes]    layer {probe.layer}  CV acc {probe.cv_accuracy}  "
              f"base rate {probe.base_rate}  n={probe.n_train}")
    print(f"[probes] done -> {out_dir}")
