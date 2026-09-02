"""nnsight-backed activation reading + patching. Used by exp2 and exp3.

Why nnsight here and TransformerLens in exp1: TransformerLens rebuilds the HF
weights into its own parameter format on load -- a ~2x transient RAM peak that
OOMs a 24 GB box even on a 2B model. `nnsight.LanguageModel` wraps the HF model
as-is and `device_map`s it straight to the GPU, so loading costs ~1x the fp16
weight size with no conversion. It also runs on any decoder that exposes
`model.model.layers[i]` -- gemma-2, Qwen2.5/3, Llama-3, Mistral -- which is what
the `--model` flag needs. exp1 stays on TransformerLens because it is bound to
the Gemma Scope SAEs (gemma-2-2b only) anyway.

Residual-stream hook point: `model.model.layers[L].output[0]` -- the hidden
state after block L, the HF/nnsight equivalent of TransformerLens's
`blocks.L.hook_resid_post`. Inside an nnsight trace, nnsight narrows the batch
dimension, so residual activations come back as `[seq, hidden]`; `model.output.
logits` stays `[batch, seq, vocab]`.
"""
from __future__ import annotations

import os
from typing import Sequence

# HF hub symlinks need admin / Developer Mode on Windows (WinError 1314); copy instead.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def pick_device_dtype(device: str | None = None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16  # Pascal: no hw bf16
    else:
        dtype = torch.float32
    return device, dtype


def load(model_id: str, device: str | None = None):
    """Dispatched `nnsight.LanguageModel` for `model_id` (downloads if needed)."""
    from nnsight import LanguageModel

    device, dtype = pick_device_dtype(device)
    return LanguageModel(
        model_id,
        device_map=device,
        dtype=dtype,
        dispatch=True,
        attn_implementation="eager",  # gemma-2 requires it; a no-op elsewhere
    )


def n_layers(model) -> int:
    return int(model.config.num_hidden_layers)


def middle_layer(model) -> int:
    return n_layers(model) // 2


def _resid(model, layer: int):
    return model.model.layers[layer].output[0]


def read(model, layer: int, input_ids):
    """`(resid[seq, hidden], logits[batch, seq, vocab])` for one prompt."""
    with model.trace(input_ids):
        resid = _resid(model, layer).save()
        logits = model.output.logits.save()
    return resid, logits


def read_resid(model, layer: int, input_ids):
    with model.trace(input_ids):
        resid = _resid(model, layer).save()
    return resid


def generate(
    model,
    input_ids,
    max_new_tokens: int,
    *,
    layer: int | None = None,
    source_resid=None,
    positions: Sequence[int] | None = None,
) -> list[int]:
    """Greedy-decode; return only the newly generated token ids.

    If `layer` + `source_resid` + `positions` are all given, splice
    `source_resid[p]` into the residual stream at each position `p` during the
    prefill pass only. Prompt-position activations are frozen into the KV cache
    after prefill, so prefill-only is both the correct scope and enough for the
    intervention to persist through the whole generation.
    """
    patch = layer is not None and source_resid is not None and positions
    pos = list(positions) if positions else []
    prompt_len = input_ids.shape[1]

    with model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False) as tracer:
        if patch:
            with tracer.iter[0]:
                hs = _resid(model, layer)
                for p in pos:
                    hs[p, :] = source_resid[p, :].to(hs.dtype)
        out = model.generator.output.save()

    return out[0][prompt_len:].tolist()


def first_token_id(tokenizer, word: str) -> int:
    """Token id the model emits first for `word` (tries a leading space first)."""
    for cand in (" " + word.strip(), word.strip()):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if ids:
            return int(ids[0])
    raise ValueError(f"could not tokenize {word!r}")


def prob_of_words(tokenizer, logits_row, words: Sequence[str]) -> dict[str, float]:
    """Softmax probability at one position for each word's first token."""
    import torch

    probs = torch.softmax(logits_row.float(), dim=-1)
    return {w: float(probs[first_token_id(tokenizer, w)]) for w in words}


def prefix_divergence(a_ids: Sequence[int], b_ids: Sequence[int]) -> int:
    """Index of the first position where two id sequences differ."""
    n = min(len(a_ids), len(b_ids))
    for i in range(n):
        if int(a_ids[i]) != int(b_ids[i]):
            return i
    return n
