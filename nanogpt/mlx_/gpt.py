import mlx.core as mx
from mlx import nn
from mlx.nn import losses
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
        self._context_length = context_length
        self.key = nn.Linear(input_size, head_size, bias=False)
        self.query = nn.Linear(input_size, head_size, bias=False)
        self.value = nn.Linear(input_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Using -1e9 instead of -inf as it is mentioned to be a better choice 
        # In the original MLX implementation of multi-head attention:
        # https://github.com/ml-explore/mlx/blob/0070e360a142dff0e40156e6c219190b2e94fa1e/python/mlx/nn/layers/transformer.py#L105
        infs = mx.array([[-1e9] * context_length] * context_length)
        self._inf_triu = mx.triu(infs, k=1)

    def __call__(self, x: mx.array) -> mx.array:  
        B, T, C = x.shape  # (T = context_length, C = input_size)
        k: mx.array = self.key(x)    # (B, T, T'), T' = head_size
        q: mx.array = self.query(x)  # (B, T, T')
        v: mx.array = self.value(x)  # (B, T, T')
        attention = q @ k.transpose(0, 2, 1) / mx.sqrt(mx.array(C))    # (B, T, T)
        attention = mx.tril(attention, k=0)
        attention += self._inf_triu[:T, :T]
        attention = mx.softmax(attention, axis=-1)
        attention = self.dropout(attention)  # regularization
        return attention @ v   # (B, T, C)


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
        self.heads = [Head(input_size=io_size, 
                           head_size=head_size, 
                           context_length=context_length, 
                           dropout=dropout) 
                      for _ in range(num_heads)]
        self.w_o = nn.Linear(io_size, io_size)  # This is W^O in the MultiHead equation in "Attention is All You Need"
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:  
        mh = mx.concatenate([head(x) for head in self.heads], axis=-1)  
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
            nn.ReLU(),  
            nn.Linear(4 * io_size, io_size),
            # The dropout is not mentioned in the original paper
            nn.Dropout(dropout)  
        )

    def __call__(self, x: mx.array):  
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


    def __call__(self, x: mx.array) -> mx.array:  
        # Layer normalization and skip-connections (residuals)
        x = self.ln1(x + self.mh(x))
        x = self.ln2(x + self.ff(x))
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
        self.positional_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.Sequential(*[TransformerBlock(io_size=embedding_size, 
                                                       context_length=context_length, 
                                                       num_heads=num_heads,   
                                                       dropout=dropout) 
                                        for _ in range(num_blocks)])
        self.lnorm = nn.LayerNorm(embedding_size)
        self.final_layer = nn.Linear(embedding_size, vocab_size)

    def __call__(self, 
                indices: mx.array, 
                targets: mx.array | None = None
                ) -> tuple[mx.array, mx.array | None]:  
        token_emb = self.token_embeddings(indices)
        pos_emb = self.positional_embeddings(indices)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.lnorm(x)
        logits: mx.array  = self.final_layer(x)
        
        loss = None
        if targets is not None:
            loss = losses.cross_entropy(logits, targets, axis=-1, reduction='mean')
        return logits, loss
    
    def generate(self, 
                 indices: mx.array,     # (B,T) tensor
                 max_new_tokens: int
                 ) -> Generator[mx.array, Any, Any]:
        for _ in range(max_new_tokens):
            context_window = indices[:, -self.context_length:]       # (B,T), where T is limited by context length
            logits, __ = self(context_window)                        # (B,T,C)
            logits = logits[:, -1, :]                                # Focus only on the last (T)ime step, becomes (B,C)
            probs = mx.softmax(logits, axis=-1)                      # (B,C)
            next_index = mx.random.categorical(mx.log(probs), num_samples=1)         # Random sampling, (B,1)
            indices = mx.concatenate([indices, next_index], axis=1)                  # (B,T+1)
            yield next_index
