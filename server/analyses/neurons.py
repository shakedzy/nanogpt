"""MLP-neuron feature finder.

Phase 3 precompute. For each MLP neuron in each transformer block, we scan
a chunk of the trained corpus and remember the top-K positions where the
neuron fired most strongly, along with surrounding token context.

The MLP block is `Linear → ReLU → Linear → Dropout`; the "neurons" are the
post-ReLU activations, dimension `4 × d_model = 512`. We hook the ReLU
module directly (`model.blocks[i].ff.ffn[1]`).

At inference time the cache is loaded once into a module-global dict and
served from memory — the disk file is only read on first request.
"""
from __future__ import annotations

import heapq
import json
import pathlib
from typing import Any

import numpy as np
import torch

from server.model_runner import load

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
TRAIN_BIN = REPO / "data" / "tokens_train.bin"
CACHE_PATH = REPO / "data" / "neurons.json"

TOP_K = 20
CONTEXT_BEFORE = 8
CONTEXT_AFTER = 8


def _capture_hook(layer_idx: int, store: dict):
    def hook(_module, _inputs, output: torch.Tensor) -> None:
        # output: (B, T, 4*d_model) — post-ReLU activations
        store[layer_idx] = output.detach().float().cpu()
    return hook


@torch.no_grad()
def precompute(num_tokens: int = 500_000, batch_size: int = 16) -> dict:
    """Scan the cached corpus once; write per-neuron top-K activating contexts."""
    r = load()
    if not TRAIN_BIN.exists():
        raise FileNotFoundError(
            f"Token cache {TRAIN_BIN} not found — run scripts.train_model first."
        )

    data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    n_tokens = min(num_tokens, len(data))
    ctx_len = r.model.context_length
    n_chunks = n_tokens // ctx_len

    # Detect neuron count from the first block's MLP
    hidden_dim = r.model.blocks[0].ff.ffn[0].out_features  # first Linear's output dim
    num_layers = len(r.model.blocks)

    # heaps[layer][neuron] = min-heap of (value, global_pos)
    heaps: list[list[list[tuple[float, int]]]] = [
        [[] for _ in range(hidden_dim)] for _ in range(num_layers)
    ]

    capture: dict[int, torch.Tensor] = {}
    handles = []
    for li, block in enumerate(r.model.blocks):
        relu = block.ff.ffn[1]
        handles.append(relu.register_forward_hook(_capture_hook(li, capture)))

    try:
        for chunk_start in range(0, n_chunks, batch_size):
            chunk_end = min(chunk_start + batch_size, n_chunks)
            batch_ids = np.stack(
                [
                    data[i * ctx_len : (i + 1) * ctx_len].astype(np.int64)
                    for i in range(chunk_start, chunk_end)
                ]
            )
            x = torch.from_numpy(batch_ids).to(r.device)
            r.model(x)

            for li in range(num_layers):
                acts = capture[li]  # (B, T, hidden)
                # max activation per (batch_elem, neuron), with its position
                max_vals, max_pos = acts.max(dim=1)  # both (B, hidden)
                for bi in range(acts.shape[0]):
                    global_offset = (chunk_start + bi) * ctx_len
                    vals_row = max_vals[bi].tolist()
                    pos_row = max_pos[bi].tolist()
                    for ni in range(hidden_dim):
                        v = vals_row[ni]
                        if v <= 0:
                            continue  # neuron didn't fire anywhere in this window
                        gp = global_offset + pos_row[ni]
                        heap = heaps[li][ni]
                        entry = (v, gp)
                        if len(heap) < TOP_K:
                            heapq.heappush(heap, entry)
                        elif v > heap[0][0]:
                            heapq.heapreplace(heap, entry)

            if (chunk_start // batch_size) % 50 == 0:
                pct = 100 * chunk_end / n_chunks
                print(f"[neurons] chunk {chunk_end}/{n_chunks} ({pct:.1f}%)", flush=True)
    finally:
        for h in handles:
            h.remove()

    print("[neurons] decoding contexts and writing cache...", flush=True)

    def context_for(global_pos: int) -> dict:
        before_ids = [int(data[i]) for i in range(max(0, global_pos - CONTEXT_BEFORE), global_pos)]
        after_ids = [
            int(data[i])
            for i in range(global_pos + 1, min(len(data), global_pos + 1 + CONTEXT_AFTER))
        ]
        token_id = int(data[global_pos])
        return {
            "token_id": token_id,
            "token_string": r.tokenizer.decode([token_id]),
            "before_text": r.tokenizer.decode(before_ids) if before_ids else "",
            "after_text": r.tokenizer.decode(after_ids) if after_ids else "",
        }

    layers_out: list[list[dict]] = []
    for li in range(num_layers):
        neurons_out: list[dict] = []
        for ni in range(hidden_dim):
            heap = heaps[li][ni]
            sorted_entries = sorted(heap, key=lambda x: -x[0])
            contexts = [
                {"value": v, "global_pos": gp, **context_for(gp)}
                for (v, gp) in sorted_entries
            ]
            max_val = contexts[0]["value"] if contexts else 0.0
            top_token = contexts[0]["token_string"] if contexts else ""
            neurons_out.append(
                {
                    "idx": ni,
                    "max_value": max_val,
                    "top_token": top_token,
                    "contexts": contexts,
                }
            )
        layers_out.append(neurons_out)

    result = {
        "metadata": {
            "num_layers": num_layers,
            "neurons_per_layer": hidden_dim,
            "top_k": TOP_K,
            "context_before": CONTEXT_BEFORE,
            "context_after": CONTEXT_AFTER,
            "num_tokens_scanned": n_chunks * ctx_len,
        },
        "layers": layers_out,
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result))
    print(
        f"[neurons] wrote {CACHE_PATH} "
        f"({num_layers} layers × {hidden_dim} neurons × top-{TOP_K})",
        flush=True,
    )
    return result


_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        if not CACHE_PATH.exists():
            raise FileNotFoundError(
                f"Neuron cache not found at {CACHE_PATH}. "
                "Run `uv run python -m scripts.precompute_neurons` first."
            )
        _cache = json.loads(CACHE_PATH.read_text())
    return _cache


def invalidate_cache() -> None:
    global _cache
    _cache = None


def get_layer_summary(layer: int) -> dict:
    cache = _load_cache()
    if not (0 <= layer < cache["metadata"]["num_layers"]):
        raise ValueError(f"layer out of range: {layer}")
    summaries = [
        {"idx": n["idx"], "max_value": n["max_value"], "top_token": n["top_token"]}
        for n in cache["layers"][layer]
    ]
    return {"layer": layer, "metadata": cache["metadata"], "neurons": summaries}


def get_neuron(layer: int, idx: int) -> dict:
    cache = _load_cache()
    if not (0 <= layer < cache["metadata"]["num_layers"]):
        raise ValueError(f"layer out of range: {layer}")
    if not (0 <= idx < cache["metadata"]["neurons_per_layer"]):
        raise ValueError(f"neuron idx out of range: {idx}")
    n = cache["layers"][layer][idx]
    return {
        "layer": layer,
        "idx": idx,
        "metadata": cache["metadata"],
        "max_value": n["max_value"],
        "top_token": n["top_token"],
        "contexts": n["contexts"],
    }
