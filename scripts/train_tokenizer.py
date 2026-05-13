"""Train a byte-level BPE tokenizer (vocab_size=2048) on the first 200k TinyStories examples.

Run once:
    uv run python -m scripts.train_tokenizer

Output: data/tokenizer.json — both training and the server load this exact file.
"""
from __future__ import annotations

import pathlib

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.processors import ByteLevel as ByteLevelPost
from tokenizers.trainers import BpeTrainer

NUM_EXAMPLES = 200_000
VOCAB_SIZE = 2048
SPECIAL_TOKENS = ["<|endoftext|>"]
OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "tokenizer.json"


def iter_texts(n: int):
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= n:
            break
        yield row["text"]


def main() -> None:
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDec()
    tokenizer.post_processor = ByteLevelPost(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevelPre.alphabet(),
        show_progress=True,
    )
    print(f"Streaming first {NUM_EXAMPLES:,} TinyStories examples...")
    tokenizer.train_from_iterator(iter_texts(NUM_EXAMPLES), trainer=trainer, length=NUM_EXAMPLES)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT_PATH))
    print(f"Saved tokenizer (vocab={tokenizer.get_vocab_size()}) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
