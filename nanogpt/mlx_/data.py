import mlx.core as mx
from typing import Literal


class Data:
    def __init__(self, 
                 array: mx.array,       # 1D array
                 split: float            # train/test split, 0 < x < 1
                 ) -> None:
        assert split < 1 and split > 0
        n = int(split * len(array))
        self._train = array[:n]
        self._test = array[n:]

    @property
    def train(self) -> mx.array: return self._train

    @property
    def test(self) -> mx.array: return self._test

    def get_batch(self, 
                  split: Literal['train', 'test'], 
                  batch_size: int, 
                  block_size: int,
                  ) -> tuple[mx.array, mx.array]:
        """
        Returns a random batch of size `batch_size` and example-length `block_size` from the dataset, along with its expected output
        """
        data = self._train if split == 'train' else self._test
        indices = [int(i) for i in mx.random.randint(0, len(data)-block_size, (batch_size,))]
        x = mx.stack([data[i : i + block_size] for i in indices])
        y = mx.stack([data[i+1 : i+1+block_size] for i in indices])
        return x, y
