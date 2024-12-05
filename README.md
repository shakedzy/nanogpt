# NanoGPT

My implementation of Andrej Karpathy's NanoGPT.

## Useful links
* Andrej Karpathy's original [NanoGPT repo](https://github.com/karpathy/nanoGPT)
* The [YouTube video](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Karpathy
* The [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper on arXiv

## Data
The `__resources__` folder contains 3 datasets:
* `tiny_shakespeare.txt`: same one used in the video
* `gutenberg_shakespeare.txt`: a copy of [The Complete Works of William Shakespeare](https://www.gutenberg.org/ebooks/100) from The Gutenberg Project
* `gutenberg_shakespeare_st.txt`: same as above, but with a special character (§) added at the beginning of each play

## Experiments
The notebook file (`notebook.ipynb`) contains several experiments:
* Training a simple Bigram Language Model (as seen on the beginning of Andrej's video)
* Training NanoGPT using a simple character-level encoder
* Training NanoGPT using OpenAI's GPT-4o encoder
* Training with and without the special § token, representing a beginning of a new play
