# NanoGPT

My implementation of Andrej Karpathy's NanoGPT. I wrote two variants of the same model, one using PyTorch and another using MLX. The models are identical as possible.

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

## Use it yourself
To rerun the experiments, you can clone the repo, and then from the repo's main directory, run:
```bash
pip install .
```