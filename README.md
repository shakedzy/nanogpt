# NanoGPT

My implementation of Andrej Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT).

## Data
The `__resources__` folder contains 3 datasets:
* `tiny_shakespeare.txt`: same one used in the video
* `gutenberg_shakespeare.txt`: a copy of [The Complete Works of William Shakespeare](https://www.gutenberg.org/ebooks/100) from The Gutenberg Project
* `gutenberg_shakespeare_st.txt`: same as above, but with a special character (§) added at the beginning of each play

## Experiments
The notebook file contains several experiments:
* Training a simple Bigram Language Model (as seen on the beginning of Andrej's video)
* Training NanoGPT using a simple character-level encoder
* Training NanoGPT using OpenAI's GPT-4o encoder
* Training with and without the special § token, representing a beginning of a new play