"""Shared activation-patching + generation helpers for phases 2 and 3.

All of this works on the TransformerLens `HookedTransformer` returned by
`loader.load_model`. Nothing here touches the llama.cpp / GGUF path.

The one idea: run a *source* prompt, cache its residual stream at a hook, then
run a *target* prompt with that residual stream spliced in at chosen token
positions and watch what the output does. If a late-stage output (the rhyme word
at the end of a line, the final answer after a chain of thought) flips to match
the source, the thing we patched was causally driving that output.
"""
from __future__ import annotations

from functools import partial
from typing import Sequence


def first_token_id(model, word: str) -> int:
    """Token id the model would emit first for `word`.

    Tries a leading space (the common mid-text case) then the bare word.
    """
    for cand in (" " + word.strip(), word.strip()):
        ids = model.to_tokens(cand, prepend_bos=False)[0]
        if len(ids) >= 1:
            return int(ids[0])
    raise ValueError(f"could not tokenize {word!r}")


def prob_of_words(model, logits_row, words: Sequence[str]) -> dict[str, float]:
    """Softmax probability at a single position for each word's first token."""
    import torch

    probs = torch.softmax(logits_row.float(), dim=-1)
    return {w: float(probs[first_token_id(model, w)]) for w in words}


def cache_resid(model, tokens, hook_name: str):
    """Return (logits, residual_stream_tensor) for `tokens` at `hook_name`."""
    import torch

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens, names_filter=hook_name, return_type="logits"
        )
    return logits, cache[hook_name]


def _splice(activation, hook, source_resid, positions):
    # source_resid: (batch, src_len, d_model) cached from the source run.
    # We only splice positions that exist in both runs.
    for p in positions:
        if p < activation.shape[1] and p < source_resid.shape[1]:
            activation[:, p, :] = source_resid[:, p, :]
    return activation


def patched_logits(model, tokens, hook_name: str, source_resid, positions: Sequence[int]):
    """Logits for `tokens` with `source_resid` spliced in at `positions`."""
    import torch

    hook = (hook_name, partial(_splice, source_resid=source_resid, positions=list(positions)))
    with torch.no_grad():
        return model.run_with_hooks(tokens, fwd_hooks=[hook], return_type="logits")


def generate_greedy(
    model,
    tokens,
    max_new_tokens: int,
    hook_name: str | None = None,
    source_resid=None,
    positions: Sequence[int] | None = None,
    stop_on_newline: bool = False,
) -> str:
    """Greedy decode. If a hook + source_resid + positions are given, the splice
    is re-applied at those (absolute, prompt-relative) positions on every step,
    so the intervention persists through the whole generation.
    """
    import torch

    fwd_hooks = []
    if hook_name and source_resid is not None and positions:
        fwd_hooks = [
            (hook_name, partial(_splice, source_resid=source_resid, positions=list(positions)))
        ]

    cur = tokens
    new_ids: list[int] = []
    newline_id = first_token_id(model, "\n") if stop_on_newline else None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = (
                model.run_with_hooks(cur, fwd_hooks=fwd_hooks, return_type="logits")
                if fwd_hooks
                else model(cur, return_type="logits")
            )
        nxt = int(logits[0, -1].argmax())
        if nxt == model.tokenizer.eos_token_id:
            break
        new_ids.append(nxt)
        if stop_on_newline and newline_id is not None and nxt == newline_id and len(new_ids) > 1:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    return model.to_string(torch.tensor(new_ids)) if new_ids else ""


def last_word(text: str) -> str:
    """Last run of alphabetic characters in `text`, lowercased. '' if none."""
    import re

    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    return words[-1].lower() if words else ""
