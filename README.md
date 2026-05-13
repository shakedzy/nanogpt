# NanoGPT

My implementation of Andrej Karpathy's NanoGPT. I wrote two variants of the same model, one using PyTorch and another using MLX. The models are identical as possible.

Also, **I've added an interpretation module**, allowing to look inside the model itslef to gain additional insights on how transformers actually work and function in the real world.

## Useful links
* Andrej Karpathy's original [NanoGPT repo](https://github.com/karpathy/nanoGPT)
* The [YouTube video](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Karpathy
* The [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper on arXiv

## Data
The `__resources__` folder contains 3 datasets:
* `tiny_shakespeare.txt`: same one used in Karpathy's video
* `gutenberg_shakespeare.txt`: a copy of [The Complete Works of William Shakespeare](https://www.gutenberg.org/ebooks/100) from The Gutenberg Project
* `gutenberg_shakespeare_st.txt`: same as above, but with a special character (§) added at the beginning of each play

## Experiments
### In file `nanogpt_experiment.ipynb`
_This file uses only the PyTorch version of the NanoGPT model_
* Training a simple Bigram Language Model (as seen on the beginning of Andrej's video)
* Training NanoGPT using a simple character-level encoder
* Training NanoGPT using OpenAI's GPT-4o encoder
* Training with and without the special § token, representing a beginning of a new play

### In file `mlx_vs_torch.ipynb`
This files contains a comparison of both NanoGPT variants (PyTorch & MLX), in training and inference time.
Both are set to use the GPU as the default device.

#### Technical details
| System | Version | 
| --- | --- |
| Computer | MacBook Pro, 14-inch, Nov 2023 |
| Chip | M3 Pro |
| Memory | 36GB |
| MacOS version | 15.1.1 |
| Python version | 3.12.3 |
| `torch` version | 2.5.1 |
| `mlx` version | 0.21.1 |

---

# NanoGPT Interp

A small web dashboard for poking at a NanoGPT-style transformer one component
at a time. The goal is to build hands-on intuition for what each piece of a
transformer is doing, and eventually find a real induction-head circuit.

**A pre-trained model exists in `server/checkpoints/tiny_stories.pt`**. The model uses the architecture in `nanogpt/torch_/gpt.py` with 6 layers, 4 heads,
`d_model=128`, `ctx_len=128` and vocab from a custom 2048-entry byte-level BPE trained on TinyStories.

## Using the pre-trained model

```bash
uv sync          # install Python deps (first run installs frontend deps too)
./run.sh         # boots backend + frontend
```

Then open <http://localhost:5173>. The backend lives on <http://127.0.0.1:8000>
and exposes `/health`, `/forward`, `/attention`, `/logit_lens`,
`/neurons/{layer}`, `/neuron/{layer}/{idx}`, `/ablate`, `/induction_scan`.
Visiting the backend root redirects to the frontend. Both the BPE tokenizer
(`data/tokenizer.json`) and the precomputed neuron-feature cache
(`data/neurons.json`) ship with the repo and are kept consistent with the
checkpoint.

## Retraining from scratch

The shipped checkpoint reaches val_loss ≈ 2.15 on TinyStories. If you want
to retrain — to experiment with different hyperparameters, fix something
in the model code, or push toward the induction-head phase transition —
follow these steps. Expect about **55 minutes of pure training on an M3
Pro** (plus a one-time ~5-minute TinyStories download on the very first
run).

1. **Delete the current checkpoint**:
   ```bash
   rm server/checkpoints/tiny_stories.pt
   ```
2. **Run training.** First-time runs auto-download the TinyStories dataset
   from HuggingFace and encode 500k stories to `data/tokens_{train,val}.bin`
   (these stay cached for subsequent runs). Training is 60k steps with
   cosine LR + AdamW, batch_size=64, ctx_len=128. The script writes the
   best-val checkpoint to `server/checkpoints/tiny_stories.pt`:
   ```bash
   uv run python -m scripts.train_model
   ```
   Useful flags (sensible defaults — only override if you know what you're
   tuning): `--train-n` (corpus size, default 200000),
   `--max-steps` (default 60000), `--batch-size` (default 64),
   `--lr` (default 3e-4), `--warmup` (default 500).
3. **Regenerate the neuron-feature cache** against the new weights.
   Without this step the Neurons tab serves stale top-K contexts:
   ```bash
   uv run python -m scripts.precompute_neurons
   ```
4. **Boot the dashboard** (or restart if it was already running so the
   new checkpoint is picked up):
   ```bash
   ./run.sh
   ```

If you also want to retrain the BPE tokenizer (e.g., to change vocab size),
run `uv run python -m scripts.train_tokenizer` *before* step 2. This
overwrites `data/tokenizer.json`; you'll need to retrain the model
afterward because token IDs will have changed.


## How attention extraction works

The existing `Head.forward` returns `softmax(QKᵀ/√C) @ V` — the attention
matrix itself never leaves the module. Rather than modify the model, the
hook in `server/analyses/attention.py` re-derives the attention pattern from
the head's own `module.key`, `module.query`, and `module.tril`. In `eval()`
mode dropout is a no-op, so the recomputed matrix matches what the head
actually applied at inference.

```python
for i, block in enumerate(model.blocks):
    for j, head in enumerate(block.mh.heads):
        head.register_forward_hook(_attn_hook(i, j, store))
```

## Reading the Neurons tab

The most abstract tab — worth a primer before clicking around.

**What a "neuron" is.** Each transformer block has an MLP shaped like
`Linear(128 → 512) → ReLU → Linear(512 → 128)`. The 512 numbers in the middle
(post-ReLU) are what we call **neurons**. Each one is a non-negative scalar
computed per-token: when the model sees a token in context, neuron *N* in
layer *L* emits some activation. Different neurons learn to detect different
things — the goal of this tab is to figure out **what each neuron detects**,
by looking at the tokens that make it fire hardest.

**The grid.** One cell per neuron, laid out 32×16 = 512 cells per layer.
Cell color (indigo intensity) is the **strongest single activation** that
neuron produced anywhere in the 500k-token corpus that was scanned.
Bright = "this neuron got loud somewhere"; pale = "this neuron never got
very excited about anything." Loud neurons are more likely to encode a
specific feature; quiet neurons are more likely noise.

**Clicking a cell** loads that neuron's **top-20 activating contexts** — the
20 corpus positions where it fired hardest. Each row is:

```
rank.  activation_value  ...before [activating_token] after...
```

The yellow-highlighted token is the one the neuron spiked on; read the 20
rows like detective evidence and pattern-match across the examples to guess
the feature.

**A worked example.** Click L0 neuron 321 and you see roughly:

```
1.  3.775   ...there was a little girl named Jane. [Jane] wanted some rice...
2.  3.609   ...a time there was a brave girl named [Jane]. She loved playing...
3.  3.482   ...feeling refreshed and [en]ergetic after his visit...
4.  3.390   ...tell her about the painter. [ Mommy] says they are very sweet...
```

Top-1 and top-2 are textbook character-name firings on "Jane" at story
openings. Top-3 fires on BPE subword "en" — unrelated. Top-4 fires on
"Mommy" in a similar position. Three typical conclusions for any neuron:

- **clean feature** ("the Jane neuron") — rare, exciting.
- **polysemantic** — fires mostly on one thing but with a long tail of
  unrelated tokens.
- **mush** — fires on a grab-bag of unrelated things. The majority.

The "most neurons are mush" finding is the motivation for sparse
autoencoders.

**Nothing is computed on click.** The dashboard doesn't re-run the model
when you click — everything was precomputed once by
`scripts/precompute_neurons.py` and written to `data/neurons.json`. The
server parses that JSON into a module-global dict on first request; each
cell-click is then a sub-millisecond dict lookup.

You **must re-run** the precompute script after retraining the model —
otherwise the top-20s reflect the old weights on the new tokens and are
misleading. There's no automatic invalidation today.

**Using the tab in practice:**

1. Pick a layer.
2. Click the brightest cells first — they have the strongest features.
3. Read the 20 contexts and try to label the neuron in your head
   ("name-after-introduction", "BPE-suffix-`ish`", "newline-before-dialog",
   "mush").
4. Switch layers and notice the shift — L0 tends to fire on token-level
   patterns (specific words, subwords); L3 on more abstract structure
   (punctuation roles, sentence positions).

There's no right answer on screen — just evidence to read.

## Reading the Induction tab

The headline test from the SPEC — *"do we have a real induction-head circuit?"*

**What the test does.** We build 25 sequences of the form
`[T₀, T₁, …, T₂₉, T₀, T₁, …, T₂₉]` — 30 random distinct tokens, then the
same 30 tokens again, total length 60. If a head has learned an
**induction** algorithm, then at every second-half position `i`, it should
look back to position `i − 29` — the token that came right after the
previous occurrence of the current token — and attend strongly to it. The
score for each head is the average attention weight it puts on that
specific induction-target cell, averaged across all second-half positions
and all 25 sequences. High = induction-like behavior.

**How to read the score.** Three reference numbers:

- ≈ 0.022 is the **uniform-attention baseline**. A head that attends
  uniformly across every legal position would land here.
- ≥ 0.4 is the **candidate threshold** from Olsson et al. (2022). A head
  scoring this high is putting most of its attention mass on exactly the
  induction-target cell — a real induction head.
- Between those, the head is *trying* but hasn't committed: there's
  structural signal but not a clean algorithm.

**The leaderboard** ranks all 24 heads (6 layers × 4 heads) descending by
score. Bars are proportional to the score. Green = above the 0.4 candidate
threshold (none in our current model); indigo = below.

**Clicking a row** loads the head's attention pattern on a sample 60-token
sequence. Lower-left triangle is what matters; upper-right is blank because
of the causal mask.

**Common attention patterns** to name when you inspect a heatmap:

- *Induction head* — a bright diagonal stripe in the lower-right quadrant,
  offset by −29 from the main diagonal (row `i` ≥ 30 attends to column
  `i − 29`). The signature we're hunting for.
- *BOS / attention sink* — a bright **vertical column at the left edge**:
  every row attends to position 0 (or the first few positions). The head
  is ignoring content and parking its attention at the start — softmax has
  to put probability mass *somewhere*, so heads often use BOS as a safe
  default. Very common, especially in deeper layers.
- *Previous-token head* — a thin diagonal stripe one cell below the main
  diagonal (row `i` attends to column `i − 1`). The structural companion
  to induction heads: layer-0 previous-token info is what later induction
  heads read from.
- *Diffuse / noise* — scattered bright cells with no clear shape. The head
  isn't doing anything interpretable on this synthetic test (could still
  be useful on real text).

**Empirical result on this checkpoint:** top head **L1H3 scores 0.0808** —
about 3.6× the uniform baseline but 1/5 of the candidate threshold. Two
heads in layer 1 (L1H3 and L1H2) are coherently elevated, which is exactly
where the canonical induction head lives (composing with a layer-0
previous-token head). This is *incipient induction* — the model is
starting to build the circuit but hasn't snapped into the canonical
pattern. Pushing further would require a bigger model or substantially
more training.