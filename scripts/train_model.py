"""Train the NanoGPT on TinyStories.

    uv run python -m scripts.train_model

On first run, streams TinyStories from HF, encodes with `data/tokenizer.json`,
and caches the result to `data/tokens_{train,val}.bin` (uint16 mmap). Then
trains a 4L/4H/d128/ctx128 NanoGPT and saves the best-val-loss checkpoint to
`server/checkpoints/tiny_stories.pt`. The interp dashboard's `model_runner`
auto-loads that file on next start.

The architecture constants here must match `server/model_runner.py`.
"""
from __future__ import annotations

import argparse
import collections
import math
import pathlib
import time

import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from nanogpt.torch_ import NanoGPT

REPO = pathlib.Path(__file__).resolve().parent.parent
TOK_PATH = REPO / "data" / "tokenizer.json"
TRAIN_BIN = REPO / "data" / "tokens_train.bin"
VAL_BIN = REPO / "data" / "tokens_val.bin"
CKPT_PATH = REPO / "server" / "checkpoints" / "tiny_stories.pt"

# Must match server/model_runner.py so the dashboard can load this checkpoint.
NUM_BLOCKS = 6
NUM_HEADS = 4
EMBED_DIM = 128
CONTEXT_LEN = 128
DROPOUT = 0.1  # dropout is a no-op once the dashboard calls .eval()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_corpus(num_train: int, num_val: int) -> None:
    if TRAIN_BIN.exists() and VAL_BIN.exists():
        print(f"[data] using cached {TRAIN_BIN.name}, {VAL_BIN.name}", flush=True)
        return
    tok = Tokenizer.from_file(str(TOK_PATH))
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        eot = 0
    total_needed = num_train + num_val
    print(
        f"[data] downloading TinyStories (need {total_needed:,} examples)",
        flush=True,
    )
    # Non-streaming: downloads parquet shards once via huggingface_hub (with its
    # own robust retry logic), then we read locally without per-row HTTP calls.
    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{total_needed}]")
    print(f"[data] loaded {len(ds):,} examples; encoding...", flush=True)
    TRAIN_BIN.parent.mkdir(parents=True, exist_ok=True)
    n_train = n_val = 0
    with open(TRAIN_BIN, "wb") as ftr, open(VAL_BIN, "wb") as fva:
        for i in tqdm(range(len(ds)), desc="encoding"):
            ids = tok.encode(ds[i]["text"]).ids
            ids.append(eot)
            arr = np.array(ids, dtype=np.uint16)
            if i < num_train:
                arr.tofile(ftr)
                n_train += len(arr)
            else:
                arr.tofile(fva)
                n_val += len(arr)
    print(f"[data] train tokens: {n_train:,}  val tokens: {n_val:,}", flush=True)


def get_batch(data: np.memmap, batch_size: int, ctx_len: int, device: torch.device):
    ix = np.random.randint(0, len(data) - ctx_len - 1, size=batch_size)
    x = np.stack([data[i : i + ctx_len].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + ctx_len].astype(np.int64) for i in ix])
    return (
        torch.from_numpy(x).to(device, non_blocking=True),
        torch.from_numpy(y).to(device, non_blocking=True),
    )


def lr_for_step(step: int, max_steps: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    encode_corpus(args.train_n, args.val_n)

    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    tok = Tokenizer.from_file(str(TOK_PATH))
    vocab = tok.get_vocab_size()
    device = pick_device()

    print(
        f"[data] train={len(train_data):,} tok  val={len(val_data):,} tok  vocab={vocab}",
        flush=True,
    )
    print(
        f"[run]  device={device}  batch_size={args.batch_size}  ctx={CONTEXT_LEN}  "
        f"max_steps={args.max_steps}",
        flush=True,
    )

    model = NanoGPT(
        vocab_size=vocab,
        embedding_size=EMBED_DIM,
        context_length=CONTEXT_LEN,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparams:,}", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(val_data, args.batch_size, CONTEXT_LEN, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    best_val = float("inf")
    start = time.time()
    model.train()
    recent_losses: collections.deque = collections.deque(maxlen=50)

    for step in range(args.max_steps + 1):
        if step % args.eval_every == 0:
            val_loss = evaluate()
            elapsed = time.time() - start
            print(
                f"[step {step:>6}/{args.max_steps}]  val_loss={val_loss:.4f}  elapsed={elapsed:.0f}s",
                flush=True,
            )
            if val_loss < best_val:
                best_val = val_loss
                CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), CKPT_PATH)
                print(f"[ckpt] saved {CKPT_PATH}  best val={best_val:.4f}", flush=True)
            if step == args.max_steps:
                break

        lr = lr_for_step(step, args.max_steps, args.warmup, args.lr, args.lr * 0.1)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, CONTEXT_LEN, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent_losses.append(loss.item())

        if step > 0 and step % args.log_every == 0:
            rolling = sum(recent_losses) / len(recent_losses)
            print(
                f"[step {step:>6}/{args.max_steps}]  train_loss={loss.item():.4f}  "
                f"train_loss_avg50={rolling:.4f}  lr={lr:.6f}",
                flush=True,
            )

    print(f"[done] best val_loss={best_val:.4f}  checkpoint at {CKPT_PATH}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=200_000)
    ap.add_argument("--val-n", type=int, default=2_000)
    ap.add_argument("--max-steps", type=int, default=15_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
