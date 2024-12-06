import torch
from torch import nn
from torch.nn import functional as F
from typing import Generator, Any


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, 
                indices: torch.Tensor, 
                targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits: torch.Tensor = self.token_embedding_table(indices)  # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_ = logits.view(B*T, C)   # torch cross-entropy expects C as the second dimension
            targets_ = targets.view(-1)     # flatten the target tensor to (B*T,)
            loss = F.cross_entropy(logits_, targets_)
        return logits, loss                 # return: (B,T,C), (B,T)

    def generate(self, 
                 indices: torch.Tensor,     # (B,T) tensor
                 max_new_tokens: int
                 ) -> Generator[torch.Tensor, Any, Any]:
        for _ in range(max_new_tokens):
            logits, __ = self(indices)                             # (B,T,C)
            logits = logits[:, -1, :]                              # Focus only on the last (T)ime step, becomes (B,C)
            probs = F.softmax(logits, dim=-1)                      # (B,C)
            next_index = torch.multinomial(probs, num_samples=1)   # Random sampling, (B,1)
            indices = torch.cat((indices, next_index), dim=1)      # (B,T+1)
            yield next_index
