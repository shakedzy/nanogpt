import torch
from typing import Literal


class Data:
    def __init__(self, 
                 tensor: torch.Tensor,   # 1D array
                 split: float            # train/test split, 0 < x < 1
                 ) -> None:
        assert split < 1 and split > 0
        n = int(split * len(tensor))
        self._train = tensor[:n]
        self._test = tensor[n:]

    @property
    def train(self) -> torch.Tensor: return self._train

    @property
    def test(self) -> torch.Tensor: return self._test

    def get_batch(self, 
                  split: Literal['train', 'test'], 
                  batch_size: int, 
                  block_size: int,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a random batch of size `batch_size` and example-length `block_size` from the dataset, along with its expected output
        """
        data = self._train if split == 'train' else self._test
        indices = torch.randint(0, len(data)-block_size, (batch_size,))
        x = torch.stack([data[i : i+block_size] for i in indices])
        y = torch.stack([data[i+1 : i+1+block_size] for i in indices])
        return x, y
