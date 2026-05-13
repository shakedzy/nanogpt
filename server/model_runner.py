"""Loads the tokenizer and NanoGPT once at startup; exposes `run(prompt)`.

Phase 0 uses random weights — the smoke test only needs token IDs to round-trip.
When `CHECKPOINT_PATH` exists on disk it will be loaded instead.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import torch
from tokenizers import Tokenizer

from nanogpt.torch_ import NanoGPT

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer.json"
CHECKPOINT_PATH = REPO_ROOT / "server" / "checkpoints" / "tiny_stories.pt"

# Model spec. Must match scripts/train_model.py exactly so trained checkpoints load.
NUM_BLOCKS = 6
NUM_HEADS = 4
EMBED_DIM = 128
CONTEXT_LEN = 128
DROPOUT = 0.0  # eval-only at inference time


@dataclass
class Runner:
    model: NanoGPT
    tokenizer: Tokenizer
    device: torch.device
    vocab_size: int


_runner: Optional[Runner] = None


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load() -> Runner:
    """Load tokenizer + model once. Idempotent."""
    global _runner
    if _runner is not None:
        return _runner

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run `uv run python -m scripts.train_tokenizer` first."
        )
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()

    device = _pick_device()
    model = NanoGPT(
        vocab_size=vocab_size,
        embedding_size=EMBED_DIM,
        context_length=CONTEXT_LEN,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    )

    if CHECKPOINT_PATH.exists():
        state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        loaded_from = str(CHECKPOINT_PATH)
    else:
        loaded_from = "random init (no checkpoint yet)"

    model.to(device).eval()
    _runner = Runner(model=model, tokenizer=tokenizer, device=device, vocab_size=vocab_size)
    print(f"[model_runner] device={device} vocab={vocab_size} weights={loaded_from}")
    return _runner


@torch.no_grad()
def run(prompt: str) -> dict:
    """Forward pass on `prompt`. Returns JSON-serializable dict with tokens + logits."""
    r = load()
    enc = r.tokenizer.encode(prompt)
    ids = enc.ids[: r.model.context_length]
    if not ids:
        ids = [0]  # avoid empty tensor; pick token 0 as a placeholder
    token_strings = [r.tokenizer.decode([i]) for i in ids]

    x = torch.tensor([ids], dtype=torch.long, device=r.device)
    logits, _ = r.model(x)  # (1, T, V)
    logits = logits[0].float().cpu()  # (T, V)

    return {
        "prompt": prompt,
        "tokens": ids,
        "token_strings": token_strings,
        "vocab_size": r.vocab_size,
        "context_length": r.model.context_length,
        "logits_shape": list(logits.shape),
        "logits": logits.tolist(),
    }
