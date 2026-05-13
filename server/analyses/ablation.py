"""Zero-ablation: run the model twice — once normally (baseline), once with
specified heads/neurons forced to zero output — and report top-K next-token
predictions for the final position from both runs.

This is "causal evidence not just correlational viewing": if a head is doing
useful work, zeroing it should shift the model's prediction; if it's
redundant, the prediction barely moves.

Hooks:
- Head ablation:   register on `model.blocks[L].mh.heads[H]` and return zeros.
- Neuron ablation: register on `model.blocks[L].ff.ffn[1]` (the ReLU) and
  zero out only the specified neuron indices in the post-ReLU activations.
"""
from __future__ import annotations

import torch

from server.model_runner import load

TOP_K = 5


def _zero_head_hook(_module, _inputs, output):
    return torch.zeros_like(output)


def _zero_neurons_hook_factory(indices: list[int]):
    idx_tensor = torch.tensor(indices, dtype=torch.long)

    def hook(_module, _inputs, output):
        out = output.clone()
        out[..., idx_tensor.to(out.device)] = 0
        return out

    return hook


def _topk_from_logits(logits: torch.Tensor, tokenizer, k: int) -> list[dict]:
    """logits: (V,) — return list of k {token_id, token_string, prob}."""
    probs = torch.softmax(logits.float(), dim=-1)
    top_p, top_ids = torch.topk(probs, k=min(k, probs.shape[-1]))
    return [
        {
            "token_id": int(tid),
            "token_string": tokenizer.decode([int(tid)]),
            "prob": float(p),
        }
        for tid, p in zip(top_ids.tolist(), top_p.tolist())
    ]


@torch.no_grad()
def compute(
    prompt: str,
    ablate_heads: list[list[int]] | None = None,
    ablate_neurons: list[list[int]] | None = None,
    top_k: int = TOP_K,
) -> dict:
    r = load()
    enc = r.tokenizer.encode(prompt)
    ids = enc.ids[: r.model.context_length]
    if not ids:
        ids = [0]
    token_strings = [r.tokenizer.decode([i]) for i in ids]
    x = torch.tensor([ids], dtype=torch.long, device=r.device)

    # Baseline forward — no hooks
    baseline_logits, _ = r.model(x)
    baseline_top = _topk_from_logits(baseline_logits[0, -1], r.tokenizer, top_k)

    # Build hooks for the ablation forward
    handles = []
    head_set: set[tuple[int, int]] = {(int(l), int(h)) for l, h in (ablate_heads or [])}
    for li, hi in head_set:
        head_mod = r.model.blocks[li].mh.heads[hi]
        handles.append(head_mod.register_forward_hook(_zero_head_hook))

    # Group neuron ablations by layer
    neuron_groups: dict[int, list[int]] = {}
    for layer_idx, neuron_idx in ablate_neurons or []:
        neuron_groups.setdefault(int(layer_idx), []).append(int(neuron_idx))
    for li, indices in neuron_groups.items():
        relu = r.model.blocks[li].ff.ffn[1]
        handles.append(relu.register_forward_hook(_zero_neurons_hook_factory(indices)))

    try:
        ablated_logits, _ = r.model(x)
    finally:
        for h in handles:
            h.remove()

    ablated_top = _topk_from_logits(ablated_logits[0, -1], r.tokenizer, top_k)

    # Also report, for each token in baseline's top-K, what probability that
    # token got under ablation. Lets the UI show "baseline's top word dropped from X% to Y%".
    ablated_probs = torch.softmax(ablated_logits[0, -1].float(), dim=-1)
    baseline_top_under_ablation = [
        {
            "token_id": entry["token_id"],
            "token_string": entry["token_string"],
            "prob": float(ablated_probs[entry["token_id"]].item()),
        }
        for entry in baseline_top
    ]

    return {
        "prompt": prompt,
        "tokens": ids,
        "token_strings": token_strings,
        "ablated_heads": sorted([list(t) for t in head_set]),
        "ablated_neurons": [
            [layer, ni] for layer, idxs in sorted(neuron_groups.items()) for ni in sorted(idxs)
        ],
        "baseline_top_k": baseline_top,
        "ablated_top_k": ablated_top,
        "baseline_top_under_ablation": baseline_top_under_ablation,
    }
