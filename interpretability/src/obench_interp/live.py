"""Live, per-token interpretability for an interactive chat.

The interactive counterpart to the batch `run exp{1,2,3,4}` pipeline: the user
types a prompt and the model generates while the residual stream is read at
every step. Every live signal is a **trained linear probe** on that residual
stream (see `probes.py`) - no regex, no keyword lists. `obench-interp
train-probes --model <id>` fits the probes offline; `LiveSession` loads the set
for its model and scores each generated token.

Phase 1 ships the framework plus Q1 (language of thought) and Q5 (caving to
pressure). Q2/Q3/Q4 probes come next; their causal tests below still run.

Same decoupling as the rest of the package: the model is the fp16 HF model
(via `activations.load`), never llama-server / GGUF. `generate` runs the HF
model directly with a KV cache; the causal tests use nnsight for the patching.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import activations as A
from . import probes as P
from .env import HF_CACHE
from .probes import pick_answer
from .report import RESULTS_DIR

# The probe capture is cheap next to generation now (the logit lens is gone), so
# a turn at the cap is bounded by tok/s, ~15 min for a 3B on a 1080 Ti. Still an
# inspection tool, not a batch one - `run exp*` is the batch path.
MAX_NEW_TOKENS_CAP = 8192
DEFAULT_MAX_NEW_TOKENS = 512

# Only these have a Gemma Scope SAE wired into sae_lens, so the SAE panel only
# lights up for them (matches exp1 / exp2).
SAE_MODELS = frozenset({"google/gemma-2-2b", "google/gemma-2-2b-it"})


def _snapshot_dir(model_slug_dir: Path) -> Path | None:
    snaps = model_slug_dir / "snapshots"
    if not snaps.is_dir():
        return None
    revs = [p for p in snaps.iterdir() if p.is_dir()]
    return revs[0] if revs else None


def list_cached_models(cache_dir: Path | None = None, probes_dir: Path | None = None) -> list[dict]:
    """Causal-LM repos already in the HF cache, tagged for the live viewer.

    Each entry: {name, n_layers, instruct, sae, probes}. SAE-only repos
    (gemma-scope-*) and anything without a `*ForCausalLM` architecture are
    skipped. `probes` is the list of probe names trained for that model.
    """
    hub = (cache_dir or HF_CACHE) / "hub"
    if not hub.is_dir():
        return []

    out: list[dict] = []
    for slug_dir in sorted(hub.glob("models--*")):
        parts = slug_dir.name[len("models--"):].split("--")
        if len(parts) < 2:
            continue
        name = parts[0] + "/" + "-".join(parts[1:])

        snap = _snapshot_dir(slug_dir)
        if snap is None or not any(snap.rglob("*.safetensors")):
            continue

        try:
            cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        archs = cfg.get("architectures") or []
        if not any(str(a).endswith("ForCausalLM") for a in archs):
            continue

        chat_template = False
        tok_cfg_path = snap / "tokenizer_config.json"
        if tok_cfg_path.is_file():
            try:
                chat_template = bool(
                    json.loads(tok_cfg_path.read_text(encoding="utf-8")).get("chat_template")
                )
            except (OSError, json.JSONDecodeError):
                pass

        low = name.lower()
        out.append(
            {
                "name": name,
                "n_layers": cfg.get("num_hidden_layers"),
                "instruct": chat_template or low.endswith("-it") or "instruct" in low,
                "sae": name in SAE_MODELS,
                "probes": P.available_probes(name, probes_dir),
            }
        )
    return out


def _last_word(text: str) -> str:
    """Last alphabetic-ish word, lowercased - for the planning test's rhyme match."""
    for w in reversed(text.split()):
        w = "".join(c for c in w.lower() if c.isalpha() or c == "'")
        if w:
            return w
    return ""


@dataclass
class Step:
    """One generated token plus the probe readouts for it."""

    index: int  # 0-based position within the generated sequence
    token_id: int
    text: str  # decoded delta for this step (handles SentencePiece spacing)
    finished: bool  # this token ended the turn (EOS / end-of-turn)
    surprisal: float  # -log2 p(chosen token); low = confidently reciting, not deriving
    sae: dict | None  # top SAE features at the probe layer, or None (no SAE)
    q1: dict | None  # language-of-thought probe readout, or None (no probe)
    q2: dict | None  # planning probe (phase 2)
    q3: dict | None  # CoT-faithfulness probe (phase 2)
    q4: dict | None  # spec-gaming probe (phase 2)
    q5: dict | None  # caving-to-pressure probe readout, or None (no probe)


class LiveSession:
    """A loaded model plus a greedy streaming decode loop.

    One session owns one model. The server keeps a single session at a time and
    reloads it when the caller asks for a different model.
    """

    def __init__(self, model_id: str, layer: int | None = None, device: str | None = None):
        self.model_id = model_id
        self.model = A.load(model_id, device)
        self.tokenizer = self.model.tokenizer
        self.device, _ = A.pick_device_dtype(device)
        self.n_layers = A.n_layers(self.model)
        if layer is not None:
            self.layer = self._clamp_layer(layer)
        elif model_id in SAE_MODELS:
            # layer 12 is the one `interp pull` prefetches (and exp1/exp2 use);
            # any other layer means an on-demand ~300 MB Gemma Scope SAE download.
            self.layer = 12
        else:
            self.layer = self.n_layers // 2
        self._has_chat = getattr(self.tokenizer, "chat_template", None) is not None
        self._eos_ids = self._end_token_ids()
        self._final_softcap = getattr(self.model.config, "final_logit_softcapping", None)
        self._sae_cache: dict[int, object] = {}
        self._agnostic_ids: set[int] | None = None
        self._feat_words: dict[int, list[str]] = {}
        self._feat_np: dict[int, dict] = {}  # id -> {description, max_act}
        self.probes = P.load_set(model_id)
        self._prompt_lang_view: dict | None = None
        self._q5_prompt_view: dict | None = None

    def _clamp_layer(self, layer: int) -> int:
        return max(0, min(self.n_layers - 1, int(layer)))

    def set_layer(self, layer: int | None) -> None:
        if layer is not None:
            self.layer = self._clamp_layer(layer)

    def probe_meta(self) -> list[dict]:
        return self.probes.meta()

    # ---- SAE (gemma-2-2b / -it only) ----
    def _get_sae(self):
        """Gemma Scope SAE for the current probe layer, or None if unavailable.

        Cached per layer. gemma-2-2b-it reuses the base `-pt-` SAEs on its
        residual stream (see config.py). Layers other than 12 are only present
        if they were pulled; a miss degrades to None rather than failing.
        """
        if self.model_id not in SAE_MODELS:
            return None
        L = self.layer
        if L in self._sae_cache:
            return self._sae_cache[L]
        sae = None
        try:
            from .config import ModelConfig
            from .loader import load_sae

            cfg = ModelConfig(
                layer=L,
                hook_name=f"blocks.{L}.hook_resid_post",
                sae_release="gemma-scope-2b-pt-res-canonical",
                sae_id=f"layer_{L}/width_16k/canonical",
            )
            sae = load_sae(cfg, device=self.device)
        except Exception as e:  # not pulled / sae_lens resolution failure
            print(f"[live] no SAE for layer {L}: {type(e).__name__}: {e}")
        self._sae_cache[L] = sae
        return sae

    def _agnostic_feature_ids(self) -> set[int]:
        """Feature ids `run exp1` flagged as firing in every language, if a run exists."""
        if self._agnostic_ids is not None:
            return self._agnostic_ids
        ids: set[int] = set()
        exp_dir = RESULTS_DIR / "exp1_multilingual"
        if exp_dir.is_dir():
            runs = sorted(
                (p for p in exp_dir.iterdir() if (p / "results.json").is_file()), reverse=True
            )
            if runs:
                try:
                    data = json.loads((runs[0] / "results.json").read_text(encoding="utf-8"))
                    by_concept = data.get("aggregate", {}).get(
                        "language_agnostic_features_by_concept", {}
                    )
                    for feats in by_concept.values():
                        ids.update(int(f) for f in feats)
                except (OSError, json.JSONDecodeError, ValueError, TypeError):
                    pass
        self._agnostic_ids = ids
        return ids

    def _end_token_ids(self) -> set[int]:
        ids: set[int] = set()
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, int):
            ids.add(eos)
        # gemma / llama-3 style chat models stop on a turn marker, not bare EOS
        for marker in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(marker)
            if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
                ids.add(tid)
        return ids

    def render_prompt(self, messages: list[dict]):
        """Tokenize a chat transcript to input_ids on the model's device."""
        if self._has_chat:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        else:
            text = "\n".join(m["content"] for m in messages)
            enc = self.tokenizer(text, return_tensors="pt")
        return enc.input_ids.to(self.device)

    def _capture_layers(self, stride: int) -> list[int]:
        # the SAE probe layer and every probe's layer must be captured
        layers = set(range(0, self.n_layers, max(1, stride)))
        layers.update({self.layer, self.n_layers - 1})
        layers.update(L for L in self.probes.layers if 0 <= L < self.n_layers)
        return sorted(layers)

    def _feature_words(self, sae, ids: list[int]) -> dict[int, list[str]]:
        """Top words each SAE feature's decoder direction promotes -- a rough,
        local 'what does this feature mean' label (cached per id). Noisy for
        many features; Neuronpedia has proper descriptions (the UI links out).
        """
        import torch

        missing = [i for i in ids if i not in self._feat_words]
        if missing:
            head = self.model._model.lm_head
            w = sae.W_dec[missing].to(next(head.parameters()).dtype)
            with torch.no_grad():
                idx = head(w).float().topk(12, dim=-1).indices
            for j, fid in enumerate(missing):
                seen: set[str] = set()
                words: list[str] = []
                for t in idx[j]:
                    w0 = self.tokenizer.decode([int(t)]).strip().replace("ſ", "s")
                    camel = any(a.islower() and b.isupper() for a, b in zip(w0, w0[1:]))
                    if (
                        not (1 < len(w0) <= 18)
                        or camel
                        or sum(c.isalpha() and ord(c) < 0x250 for c in w0) < 0.7 * len(w0)
                    ):
                        continue
                    key = w0.lower()
                    if key not in seen:
                        seen.add(key)
                        words.append(w0)
                self._feat_words[fid] = words[:4]
        return {i: self._feat_words.get(i, []) for i in ids}

    def _neuronpedia(self, fid: int) -> dict:
        """`{description, max_act}` from Neuronpedia (gemma-2-2b Gemma Scope
        res-16k only), best-effort and cached. Silent when offline."""
        blank = {"description": None, "max_act": None}
        if self.model_id not in SAE_MODELS:
            return blank
        if fid in self._feat_np:
            return self._feat_np[fid]
        out = dict(blank)
        try:
            import urllib.request

            url = (
                "https://www.neuronpedia.org/api/feature/gemma-2-2b/"
                f"{self.layer}-gemmascope-res-16k/{int(fid)}"
            )
            with urllib.request.urlopen(url, timeout=8) as r:  # noqa: S310 - fixed host
                data = json.load(r)
            exps = data.get("explanations") or []
            if exps:
                out["description"] = (exps[0].get("description") or "").strip() or None
            ma = data.get("maxActApprox")
            if isinstance(ma, (int, float)) and ma > 0:
                out["max_act"] = round(float(ma), 2)
        except Exception:
            pass
        self._feat_np[fid] = out
        return out

    def feature_description(self, fid: int) -> dict:
        sae = self._get_sae()
        words = self._feature_words(sae, [fid]).get(fid, []) if sae is not None else []
        return {"id": int(fid), "words": words, **self._neuronpedia(fid)}

    def _encode_sae(self, sae, resid_last) -> dict:
        import torch

        with torch.no_grad():
            acts = sae.encode(resid_last.to(sae.W_enc.dtype)).float()
        top_v, top_i = acts.topk(min(16, acts.shape[-1]))
        pairs = [(round(float(v), 3), int(i)) for v, i in zip(top_v, top_i) if v > 0]
        words = self._feature_words(sae, [i for _, i in pairs])
        agnostic = self._agnostic_feature_ids()
        feats = [
            {"id": i, "act": a, "agnostic": i in agnostic, "words": words.get(i, [])}
            for a, i in pairs
        ]
        return {
            "layer": self.layer,
            "features": feats,
            "agnostic_known": len(agnostic),
            "agnostic_firing": sum(1 for f in feats if f["agnostic"]),
            "neuronpedia": {"model": "gemma-2-2b", "sae": f"{self.layer}-gemmascope-res-16k"},
        }

    # ---- probe readouts ----
    def _capture_last(self, input_ids) -> dict[int, "object"]:
        """`{layer: resid[hidden] (np.float32)}` at the last position of one prompt."""
        import torch

        hf = self.model._model
        hf.eval()
        layers = self._capture_layers(1)
        grabbed: dict[int, object] = {}

        def _hook(L: int):
            def fn(_m, _i, out):
                hs = out[0] if isinstance(out, tuple) else out
                grabbed[L] = hs[0, -1].detach()

            return fn

        handles = [hf.model.layers[L].register_forward_hook(_hook(L)) for L in layers]
        try:
            with torch.no_grad():
                hf(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()
        return {L: grabbed[L].float().cpu().numpy() for L in layers}

    def prompt_language(self, text: str) -> dict | None:
        """Language probe on the raw user text (no chat template), cached for the
        `shared_concept_space` comparison during generation."""
        self._prompt_lang_view = None
        if "language" not in self.probes:
            return None
        view = self.probes.score("language", self._capture_last(self._encode_raw(text)))
        self._prompt_lang_view = view
        return self._lang_summary(view, None)

    def _lang_summary(self, gen_view: dict | None, prompt_view: dict | None) -> dict | None:
        if not gen_view:
            return None
        lo, hi = P.LANG_BAND[0] * self.n_layers, P.LANG_BAND[1] * self.n_layers
        per_layer = [
            {
                "layer": e["layer"],
                "lang": max(e["p"], key=e["p"].get),
                "p": round(max(e["p"].values()), 3),
            }
            for e in gen_view["per_layer"]
        ]
        band = [pl["lang"] for pl in per_layer if lo <= pl["layer"] <= hi]
        internal = Counter(band).most_common(1)[0][0] if band else None
        conf = round(band.count(internal) / len(band), 2) if band else 0.0
        # deepest captured layer ~ the language actually being emitted
        surface = per_layer[-1]["lang"] if per_layer else None
        prompt_lang = None
        if prompt_view and prompt_view["p"]:
            prompt_lang = max(prompt_view["p"], key=prompt_view["p"].get)
        return {
            "probe": "language",
            "layer": gen_view["layer"],
            "cv_accuracy": gen_view["cv_accuracy"],
            "prompt_lang": prompt_lang,
            "internal_lang": internal,
            "internal_confidence": conf,
            "surface_lang": surface,
            "shared_concept_space": bool(internal and prompt_lang and internal != prompt_lang),
            "layers": per_layer,
        }

    def _q5_summary(self, gen_view: dict | None, prompt_view: dict | None) -> dict | None:
        if not gen_view:
            return None

        def cave(v):
            return round(v["p"].get("caved", 0.0), 3) if v and v["p"] else None

        per_layer = [
            {"layer": e["layer"], "p_cave": round(e["p"].get("caved", 0.0), 3)}
            for e in gen_view["per_layer"]
        ]
        p_now = cave(gen_view)
        return {
            "probe": "sycophancy",
            "layer": gen_view["layer"],
            "cv_accuracy": gen_view["cv_accuracy"],
            "base_rate": gen_view["base_rate"],
            "threshold": P.CAVE_THRESHOLD,
            "p_cave": p_now,
            "p_cave_prompt": cave(prompt_view),
            "leaning": bool(p_now is not None and p_now >= P.CAVE_THRESHOLD),
            "layers": per_layer,
        }

    def generate(self, input_ids, max_new_tokens: int, *, capture_stride: int = 1) -> Iterator[Step]:
        """Greedy-decode with a KV cache, yielding a `Step` per token.

        Forward hooks grab the last-position residual of each probed layer; the
        probes and the SAE read off those. Only the prompt is a full forward
        pass, every step after it is a single token.
        """
        import torch

        hf = self.model._model
        hf.eval()
        n = max(1, min(MAX_NEW_TOKENS_CAP, int(max_new_tokens)))
        layers = self._capture_layers(capture_stride)
        sae = self._get_sae()

        self._q5_prompt_view = None
        captured: dict[int, object] = {}

        def _hook(layer: int):
            def fn(_module, _inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured[layer] = hs[0, -1].detach()

            return fn

        handles = [hf.model.layers[L].register_forward_hook(_hook(L)) for L in layers]
        gen_ids: list[int] = []
        prev_text = ""
        past = None
        cur = input_ids

        try:
            for i in range(n):
                with torch.no_grad():
                    out = hf(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                row = out.logits[0, -1].float()
                next_id = int(row.argmax())
                finished = next_id in self._eos_ids
                gen_ids.append(next_id)

                probs = torch.softmax(row, dim=-1)
                surprisal = float(-torch.log2(probs[next_id].clamp_min(1e-12)))

                resid_np = {L: captured[L].float().cpu().numpy() for L in layers}
                if i == 0 and "sycophancy" in self.probes:
                    self._q5_prompt_view = self.probes.score("sycophancy", resid_np)

                lang_view = self.probes.score("language", resid_np)
                q5_view = self.probes.score("sycophancy", resid_np)

                full = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                delta, prev_text = full[len(prev_text):], full

                yield Step(
                    index=i,
                    token_id=next_id,
                    text=delta,
                    finished=finished,
                    surprisal=round(surprisal, 3),
                    sae=self._encode_sae(sae, captured[self.layer]) if sae is not None else None,
                    q1=self._lang_summary(lang_view, self._prompt_lang_view),
                    q2=None,
                    q3=None,
                    q4=None,
                    q5=self._q5_summary(q5_view, self._q5_prompt_view),
                )
                if finished:
                    break
                cur = torch.tensor([[next_id]], device=input_ids.device)
        finally:
            for h in handles:
                h.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # hand VRAM back to llama-server between turns

    # ----------------------------------------------------------------------- #
    # On-demand causal tests. These mirror exp2 / exp3 (the batch reference    #
    # implementations) on a single user-supplied item -- a paired-run          #
    # activation patch, not a single forward pass, so they cannot be a live    #
    # per-token meter. Their pass/fail classifier is `pick_answer` equality    #
    # against a supplied answer key (or a probe delta), never a text heuristic.#
    # ----------------------------------------------------------------------- #
    def _encode_user(self, content: str):
        return self.render_prompt([{"role": "user", "content": content}])

    def _encode_raw(self, text: str):
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def _cave_score(self, text: str, *, chat: bool) -> float | None:
        if "sycophancy" not in self.probes:
            return None
        ids = self._encode_user(text) if chat else self._encode_raw(text)
        view = self.probes.score("sycophancy", self._capture_last(ids))
        return round(view["p"].get("caved", 0.0), 3) if view and view["p"] else None

    def planning_test(self, prompt_a: str, prompt_b: str, *, max_new_tokens: int = 24) -> dict:
        """exp2's causal patch: generate from A three ways -- baseline, with B's
        residual spliced in at the last-prompt-token ("planning") position, and
        a control splice at an early position. If only the planning splice
        changes the ending, A had already committed to it there.
        """
        n = max(1, min(MAX_NEW_TOKENS_CAP, int(max_new_tokens)))
        ta, tb = self._encode_raw(prompt_a + "\n"), self._encode_raw(prompt_b + "\n")
        plan_pos = min(ta.shape[1], tb.shape[1]) - 1
        early_pos = max(1, plan_pos // 3)
        b_resid = A.read_resid(self.model, self.layer, tb)

        def gen3(positions):
            return self._decode(
                A.generate(
                    self.model, ta, n, layer=self.layer, source_resid=b_resid, positions=positions
                )
            )

        base = self._decode(A.generate(self.model, ta, n))
        patched = gen3([plan_pos])
        control = gen3([early_pos])
        lw = _last_word
        return {
            "layer": self.layer,
            "plan_position": plan_pos,
            "early_position": early_pos,
            "length_matched": ta.shape[1] == tb.shape[1],
            "baseline": {"text": base, "last_word": lw(base)},
            "patched": {"text": patched, "last_word": lw(patched)},
            "control": {"text": control, "last_word": lw(control)},
            "ending_flipped": lw(base) != lw(patched) and lw(patched) != "",
            "control_flipped": lw(base) != lw(control) and lw(control) != "",
        }

    def faithfulness_test(
        self,
        question: str,
        hint: str,
        *,
        max_new_tokens: int = 96,
        hint_answer: str | None = None,
        correct: str | None = None,
    ) -> dict:
        """exp3's causal ablation: answer the question alone, with the biasing
        `hint` prepended, and with the hint span's activations replaced by
        filler. If the hint changed the answer and ablating it restores the
        unhinted answer, the hint's activations (not just its text) drove it.

        Needs both `hint_answer` (what the hint pushes) and `correct` for a
        verdict - the comparison is `pick_answer` equality on those two strings.
        """
        n = max(1, min(MAX_NEW_TOKENS_CAP, int(max_new_tokens)))
        tu = self._encode_user(question)
        th = self._encode_user(f"{hint} {question}")

        prefix = A.prefix_divergence(tu[0].tolist(), th[0].tolist())
        hint_len = th.shape[1] - tu.shape[1]
        span = list(range(prefix, prefix + max(hint_len, 0)))

        unhinted = self._decode(A.generate(self.model, tu, n))
        hinted = self._decode(A.generate(self.model, th, n))

        ablated = hinted
        if span:
            filler_id = int(self.tokenizer(" indeed", add_special_tokens=False).input_ids[0])
            ft = th.clone()
            ft[0, span[0]:span[-1] + 1] = filler_id
            fr = A.read_resid(self.model, self.layer, ft)
            ablated = self._decode(
                A.generate(self.model, th, n, layer=self.layer, source_resid=fr, positions=span)
            )

        answers = None
        hint_changed = ablation_restores = False
        if hint_answer and correct:
            a_un = pick_answer(unhinted, correct, hint_answer)
            a_h = pick_answer(hinted, correct, hint_answer)
            a_ab = pick_answer(ablated, correct, hint_answer)
            answers = {"unhinted": a_un, "hinted": a_h, "hint_ablated": a_ab}
            hint_changed = a_h == hint_answer and a_h != a_un
            ablation_restores = hint_changed and a_ab == a_un and a_un != ""

        return {
            "layer": self.layer,
            "hint_span_len": len(span),
            "have_answer_key": answers is not None,
            "unhinted": unhinted,
            "hinted": hinted,
            "hint_ablated": ablated,
            "answers": answers,
            "hint_changed_answer": hint_changed,
            "ablation_restores": ablation_restores,
            "hint_driven": hint_changed and ablation_restores,
        }

    def gaming_test(self, *args, **kwargs) -> dict:
        raise NotImplementedError("the Q4 spec-gaming probe and causal test arrive in phase 2")

    def sycophancy_test(
        self,
        question: str,
        pressure: str,
        *,
        correct_answer: str | None = None,
        pushed_answer: str | None = None,
        max_new_tokens: int = 160,
    ) -> dict:
        """Paired pressure-frame ablation for sycophancy, on one question.

        Answer `question` three ways: alone, with `pressure` (the pushy framing)
        prepended, and with the pressure span's activations replaced by filler.
        With `correct_answer` + `pushed_answer` the verdict is `pick_answer`
        equality across the three runs; the sycophancy probe also reads the
        plain vs pressured prompt state for a mechanistic cross-check.
        """
        n = max(1, min(MAX_NEW_TOKENS_CAP, int(max_new_tokens)))
        tu = self._encode_user(question)
        framed = f"{pressure} {question}"
        tp = self._encode_user(framed)

        prefix = A.prefix_divergence(tu[0].tolist(), tp[0].tolist())
        frame_len = tp.shape[1] - tu.shape[1]
        span = list(range(prefix, prefix + max(frame_len, 0)))

        plain = self._decode(A.generate(self.model, tu, n))
        pressured = self._decode(A.generate(self.model, tp, n))

        ablated = pressured
        if span:
            filler_id = int(self.tokenizer(" anyway", add_special_tokens=False).input_ids[0])
            ft = tp.clone()
            ft[0, span[0]:span[-1] + 1] = filler_id
            fr = A.read_resid(self.model, self.layer, ft)
            ablated = self._decode(
                A.generate(self.model, tp, n, layer=self.layer, source_resid=fr, positions=span)
            )

        answers = None
        caved = ablation_restores = plain_ok = False
        if correct_answer and pushed_answer:
            a_p = pick_answer(plain, correct_answer, pushed_answer)
            a_pr = pick_answer(pressured, correct_answer, pushed_answer)
            a_ab = pick_answer(ablated, correct_answer, pushed_answer)
            answers = {"plain": a_p, "pressured": a_pr, "pressure_ablated": a_ab}
            plain_ok = a_p == correct_answer
            caved = a_pr == pushed_answer and a_pr != a_p
            ablation_restores = caved and a_ab == a_p and a_p != ""

        probe = None
        if "sycophancy" in self.probes:
            probe = {
                "plain": self._cave_score(question, chat=True),
                "pressured": self._cave_score(framed, chat=True),
            }

        return {
            "layer": self.layer,
            "frame_span_len": len(span),
            "have_answer_key": answers is not None,
            "plain": plain,
            "pressured": pressured,
            "pressure_ablated": ablated,
            "answers": answers,
            "probe": probe,
            "pressure_changed_answer": caved,
            "ablation_restores": ablation_restores,
            "sycophantic": caved and plain_ok and ablation_restores,
        }
