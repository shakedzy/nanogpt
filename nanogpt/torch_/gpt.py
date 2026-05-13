import torch
from torch import nn
from torch.nn import functional as F
from typing import Generator, Any


class Head(nn.Module):
    def __init__(self, 
                 *, 
                 input_size: int, 
                 head_size: int, 
                 context_length: int,
                 dropout: float):
        assert dropout >= 0 and dropout < 1
        super().__init__()
        self.key = nn.Linear(input_size, head_size, bias=False)
        self.query = nn.Linear(input_size, head_size, bias=False)
        self.value = nn.Linear(input_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # (T = context_length, C = input_size)
        k: torch.Tensor = self.key(x)    # (B, T, H), H = head_size
        q: torch.Tensor = self.query(x)  # (B, T, H)
        v: torch.Tensor = self.value(x)  # (B, T, H)
        # The lines below are the famous attention equation, along with masking and regularization.
        h = k.size(-1)
        attention = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(h, dtype=q.dtype, device=q.device))    # (B, T, T)
        attention = attention.masked_fill(self.tril[:T,:T] == 0, -float('inf'))  # not allowing future information to flow backwards
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)  # regularization
        return attention @ v   # (B, T, H)


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 *, 
                 io_size: int,
                 context_length: int,
                 num_heads: int,
                 dropout: float):
        assert io_size % num_heads == 0
        super().__init__()
        head_size = io_size // num_heads
        self.heads = nn.ModuleList([Head(input_size=io_size, 
                                         head_size=head_size, 
                                         context_length=context_length, 
                                         dropout=dropout) 
                                    for _ in range(num_heads)])
        self.w_o = nn.Linear(io_size, io_size)  # This is W^O in the MultiHead equation in "Attention is All You Need"
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        mh = torch.cat([head(x) for head in self.heads], dim=-1)  
        mh = self.w_o(mh) 
        return self.dropout(mh)   


class FeedForward(nn.Module):
    def __init__(self, 
                 *,
                 io_size: int,
                 dropout: float):
        assert dropout >= 0 and dropout < 1
        super().__init__()
        self.ffn = nn.Sequential(
            # The three lines below are exactly Equation (2) of "Attention is All You Need"
            # The 4 scale-factor comes from the remarks under that equation
            nn.Linear(io_size, 4 * io_size),
            # GELU instead of the paper's ReLU. GELU is smooth everywhere
            # (so no "dead neurons" with permanently-zero gradient) and can
            # output slightly negative values for slightly-negative inputs,
            # giving each neuron one extra mode of expression. Used in
            # GPT-2 onwards; small but reliable improvement over ReLU.
            nn.GELU(),
            nn.Linear(4 * io_size, io_size),
            # The dropout is not mentioned in the original paper
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):  
        return self.ffn(x)   
    

class TransformerBlock(nn.Module):
    def __init__(self, 
                 *,
                 io_size: int,
                 context_length: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.mh = MultiHeadAttention(io_size=io_size, 
                                      context_length=context_length, 
                                      num_heads=num_heads,   
                                      dropout=dropout)
        self.ff = FeedForward(io_size=io_size, 
                              dropout=dropout)
        self.ln1 = nn.LayerNorm(io_size)   
        self.ln2 = nn.LayerNorm(io_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: LayerNorm goes INSIDE the residual branch, so the residual
        # stream itself is never normalized on the bypass path. "Attention is
        # All You Need" originally used post-LN — LN(x + sublayer(x)) — but
        # later work (and practice from GPT-2 onwards) showed post-LN
        # destabilizes training for deep/small transformers and needs careful
        # LR warmup. Pre-LN keeps gradients flowing back unchanged across the
        # skip connection, trains more stably, and is also what makes the
        # "residual stream as a shared workspace" interpretation valid.
        x = x + self.mh(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    

class NanoGPT(nn.Module):
    def __init__(self,
                 *,
                 vocab_size: int,
                 embedding_size: int,
                 context_length: int,
                 num_heads: int,
                 num_blocks: int,
                 dropout: float):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.positional_embeddings = nn.Embedding(context_length, embedding_size)
        self.blocks = nn.Sequential(*[TransformerBlock(io_size=embedding_size, 
                                                       context_length=context_length, 
                                                       num_heads=num_heads,   
                                                       dropout=dropout) 
                                        for _ in range(num_blocks)])
        self.lnorm = nn.LayerNorm(embedding_size)
        self.final_layer = nn.Linear(embedding_size, vocab_size)
        # Weight tying: share the (vocab_size, embedding_size) matrix between
        # the input embedding lookup and the output LM head. Row i of that
        # matrix describes "token i as a residual-stream vector" — used for
        # encoding on the input side and for scoring on the output side.
        # Both jobs are dual aspects of the same question, so reusing one
        # matrix (a) saves vocab_size * embedding_size parameters, (b) gives
        # each row twice the gradient signal (from both input and output
        # contexts), and (c) biases the model toward genuinely meaningful
        # token representations. Standard since GPT-2; see Press & Wolf 2016.
        self.final_layer.weight = self.token_embeddings.weight

    def forward(self, 
                indices: torch.Tensor, 
                targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:  
        B, T = indices.shape
        token_emb = self.token_embeddings(indices)
        pos_emb = self.positional_embeddings(torch.arange(T, device=indices.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.lnorm(x)
        logits: torch.Tensor  = self.final_layer(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_ = logits.view(B*T, C)   # torch cross-entropy expects C as the second dimension
            targets_ = targets.view(-1)     # flatten the target tensor to (B*T,)
            loss = F.cross_entropy(logits_, targets_)
        return logits, loss
    
    def generate(self, 
                 indices: torch.Tensor,     # (B,T) tensor
                 max_new_tokens: int
                 ) -> Generator[torch.Tensor, Any, Any]:
        for _ in range(max_new_tokens):
            context_window = indices[:, -self.context_length:]       # (B,T), where T is limited by context length
            logits, __ = self(context_window)                        # (B,T,C)
            logits = logits[:, -1, :]                                # Focus only on the last (T)ime step, becomes (B,C)
            probs = F.softmax(logits, dim=-1)                        # (B,C)
            next_index = torch.multinomial(probs, num_samples=1)     # Random sampling, (B,1)
            indices = torch.cat((indices, next_index), dim=1)        # (B,T+1)
            yield next_index
