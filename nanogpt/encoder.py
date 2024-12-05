import tiktoken
from abc import ABC, abstractmethod


class Encoder(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError()
    
    @abstractmethod
    def decode(self, encoded_text: list[int]) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def __len__(self):
        raise NotImplementedError()
    

class CharacterLevelEncoder(Encoder):
    """
    A character level encoder.
    """
    def __init__(self, input_text: str) -> None:
        super().__init__()
        self._chars: list[str] = sorted(list(set(input_text)))
        self._encoder: dict[str, int] = {c: i for (i, c) in enumerate(self._chars)}
        self._decoder: dict[int, str] = {i: c for (i, c) in enumerate(self._chars)}
    
    def encode(self, text: str) -> list[int]:
        return [self._encoder[c] for c in text]
    
    def decode(self, encoded_text: list[int]) -> str:
        return ''.join([self._decoder[i] for i in encoded_text])
    
    def __len__(self):
        return len(self._chars)


class TiktokenBasedEncoder(Encoder):
    """
    A simplified encoder based on OpenAI's `tiktoken` library
    """
    def __init__(self, input_text: str) -> None:
        super().__init__()
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')
        self._all_tokens = list(set(self.tokenizer.encode(input_text)))
        self._encoder: dict[int, int] = {c: i for (i, c) in enumerate(self._all_tokens)}
        self._decoder: dict[int, int] = {i: c for (i, c) in enumerate(self._all_tokens)}

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text)
        return [self._encoder[c] for c in tokens]
    
    def decode(self, encoded_text: list[int]) -> str:
        tokens = [self._decoder[i] for i in encoded_text]
        return self.tokenizer.decode(tokens)
    
    def __len__(self):
        return len(self._all_tokens)
    