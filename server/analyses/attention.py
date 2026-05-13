"""Per-head attention patterns.

We register a forward_hook on every `model.blocks[i].mh.heads[j]` and re-derive
the post-softmax (B, T, T) attention matrix from the head's own K, Q, and
causal mask. We can't read it out of the existing `Head.forward` directly —
that method only returns `attn @ V` — so we recompute. In eval() mode dropout
is a no-op, so the recomputed matrix matches what the head actually applied.
"""
from __future__ import annotations

import torch

from server.model_runner import load


def _attn_hook(layer_idx: int, head_idx: int, store: dict):
    def hook(module, inputs, _output):
        x = inputs[0]
        _, T, _ = x.shape
        k = module.key(x)
        q = module.query(x)
        h = k.size(-1)  # match Head.forward
        attn = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(h, dtype=q.dtype, device=q.device))
        attn = attn.masked_fill(module.tril[:T, :T] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        store[(layer_idx, head_idx)] = attn[0].detach().float().cpu().tolist()

    return hook


@torch.no_grad()
def compute(prompt: str) -> dict:
    r = load()
    enc = r.tokenizer.encode(prompt)
    ids = enc.ids[: r.model.context_length]
    if not ids:
        ids = [0]
    token_strings = [r.tokenizer.decode([i]) for i in ids]
    x = torch.tensor([ids], dtype=torch.long, device=r.device)

    store: dict[tuple[int, int], list] = {}
    handles = []
    for i, block in enumerate(r.model.blocks):
        for j, head in enumerate(block.mh.heads):
            handles.append(head.register_forward_hook(_attn_hook(i, j, store)))
    try:
        r.model(x)
    finally:
        for h in handles:
            h.remove()

    num_layers = len(r.model.blocks)
    num_heads = len(r.model.blocks[0].mh.heads)
    attention = [[store[(i, j)] for j in range(num_heads)] for i in range(num_layers)]

    return {
        "prompt": prompt,
        "tokens": ids,
        "token_strings": token_strings,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": len(ids),
        "attention": attention,
    }
