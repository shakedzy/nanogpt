"""Logit lens (Nostalgebraist, 2020).

For each transformer block, snapshot the residual stream at its output, then
project it through the model's own final layer norm + LM head and softmax. The
result is the vocabulary distribution that *would* come out if the model
stopped computing at that block.

Hooks attach to `model.blocks[i]` (each item in the Sequential). The block's
forward returns the residual stream after the block has updated it, which is
exactly what we want.

Note: the last layer's lens row equals the model's actual prediction, since
the model's tail after the final block is just `lnorm` + `final_layer`.
"""
from __future__ import annotations

import torch

from server.model_runner import load


@torch.no_grad()
def compute(prompt: str, top_k: int = 5) -> dict:
    r = load()
    enc = r.tokenizer.encode(prompt)
    ids = enc.ids[: r.model.context_length]
    if not ids:
        ids = [0]
    token_strings = [r.tokenizer.decode([i]) for i in ids]
    x = torch.tensor([ids], dtype=torch.long, device=r.device)

    residuals: dict[int, torch.Tensor] = {}
    handles = []
    for i, block in enumerate(r.model.blocks):

        def make_hook(idx: int):
            def hook(_module, _inputs, output):
                residuals[idx] = output.detach()
            return hook

        handles.append(block.register_forward_hook(make_hook(i)))

    try:
        r.model(x)
    finally:
        for h in handles:
            h.remove()

    num_layers = len(r.model.blocks)
    k = min(top_k, r.vocab_size)
    layers = []
    for i in range(num_layers):
        residual = residuals[i]                   # (1, T, C)
        normalized = r.model.lnorm(residual)
        logits = r.model.final_layer(normalized)  # (1, T, V)
        probs = torch.softmax(logits[0].float(), dim=-1)  # (T, V)
        top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
        rows = []
        for t in range(top_probs.shape[0]):
            cells = []
            for j in range(k):
                tid = int(top_ids[t, j].item())
                cells.append(
                    {
                        "token_id": tid,
                        "token_string": r.tokenizer.decode([tid]),
                        "prob": float(top_probs[t, j].item()),
                    }
                )
            rows.append(cells)
        layers.append(rows)

    return {
        "prompt": prompt,
        "tokens": ids,
        "token_strings": token_strings,
        "num_layers": num_layers,
        "seq_len": len(ids),
        "top_k": k,
        # layers[layer][pos] -> list of top-k {token_id, token_string, prob}
        "layers": layers,
    }
