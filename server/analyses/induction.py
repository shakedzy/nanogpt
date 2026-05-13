"""Induction-head detector.

Generates batches of random repeated sequences of the form
`[T0, T1, ..., T_{N-1}, T0, T1, ..., T_{N-1}]`, runs the model with attention
hooks, and scores every (layer, head) by how strongly it attends to the
"token that came after the previous occurrence of this token". That's the
defining behavior of an induction head per Olsson et al. (2022).

For each second-half position i (i ∈ [N, 2N-1]), the model has seen the same
token before at position i-N; the token after that previous occurrence was
at position i-N+1. So the induction score is

    score(L, H) = mean over (batch, i) of attn[L, H, i, i-N+1]

Random baseline is ~1/i (uniform softmax over legal causal sources); for our
N=30, the baseline averages to roughly 0.02. Anything above 0.4 is a strong
induction-head candidate.
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
        h = k.size(-1)
        attn = q @ k.transpose(-2, -1) / torch.sqrt(
            torch.tensor(h, dtype=q.dtype, device=q.device)
        )
        attn = attn.masked_fill(module.tril[:T, :T] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        # keep (B, T, T) — average over B later
        store[(layer_idx, head_idx)] = attn.detach().float().cpu()

    return hook


@torch.no_grad()
def compute(num_seqs: int = 25, seq_len: int = 30, seed: int = 1337) -> dict:
    r = load()
    model = r.model

    if 2 * seq_len > model.context_length:
        raise ValueError(
            f"seq_len={seq_len} (repeated → {2*seq_len}) exceeds model ctx={model.context_length}"
        )

    # Generate random sequences of distinct tokens (skip EOT id 0).
    g = torch.Generator(device="cpu").manual_seed(seed)
    halves = []
    for _ in range(num_seqs):
        perm = torch.randperm(r.vocab_size - 1, generator=g)[:seq_len] + 1
        halves.append(perm)
    half = torch.stack(halves)               # (num_seqs, seq_len)
    full = torch.cat([half, half], dim=1)     # (num_seqs, 2*seq_len)
    x = full.to(r.device)

    num_layers = len(model.blocks)
    num_heads = len(model.blocks[0].mh.heads)

    store: dict[tuple[int, int], torch.Tensor] = {}
    handles = []
    for li, block in enumerate(model.blocks):
        for hi, head in enumerate(block.mh.heads):
            handles.append(head.register_forward_hook(_attn_hook(li, hi, store)))

    try:
        model(x)
    finally:
        for h in handles:
            h.remove()

    # Score every head
    i_vals = torch.arange(seq_len, 2 * seq_len)        # query positions in second half
    k_vals = i_vals - seq_len + 1                       # corresponding induction-key positions

    scores = []
    for li in range(num_layers):
        for hi in range(num_heads):
            attn = store[(li, hi)]  # (num_seqs, 2N, 2N)
            # gather attn[:, i_vals, k_vals]
            gathered = attn[:, i_vals, k_vals]          # (num_seqs, N)
            scores.append(
                {
                    "layer": li,
                    "head": hi,
                    "score": float(gathered.mean()),
                }
            )

    scores_sorted = sorted(scores, key=lambda s: -s["score"])

    # One sample sequence's full per-head attention for the inspector UI.
    sample = {
        "tokens": full[0].tolist(),
        # attention[L][H][q][k]
        "attention": [
            [store[(li, hi)][0].tolist() for hi in range(num_heads)]
            for li in range(num_layers)
        ],
    }

    return {
        "num_seqs": num_seqs,
        "seq_len": seq_len,
        "total_len": 2 * seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "candidate_threshold": 0.4,
        "scores": scores_sorted,
        "sample": sample,
    }
