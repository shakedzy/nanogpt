"""Precompute top-K activating contexts for every MLP neuron in every layer.

    uv run python -m scripts.precompute_neurons

Reads `data/tokens_train.bin` (the cached training tokens), runs the current
checkpoint over it, and writes `data/neurons.json`. Must be re-run after the
model is retrained — otherwise the cache reflects the wrong weights.
"""
from __future__ import annotations

import argparse

from server.analyses.neurons import precompute


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num-tokens",
        type=int,
        default=500_000,
        help="how many tokens of the train corpus to scan (capped at corpus size)",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    result = precompute(num_tokens=args.num_tokens, batch_size=args.batch_size)
    meta = result["metadata"]
    print(
        f"done: {meta['num_layers']} layers × {meta['neurons_per_layer']} neurons, "
        f"top-{meta['top_k']} contexts each, scanned {meta['num_tokens_scanned']:,} tokens"
    )


if __name__ == "__main__":
    main()
