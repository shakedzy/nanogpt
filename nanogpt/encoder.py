class Encoder:
    """
    A character level encoder.
    """
    def __init__(self, input_text: str) -> None:
        self._chars: list[str] = sorted(list(set(input_text)))
        self._encoder: dict[str, int] = {c: i for (i, c) in enumerate(self._chars)}
        self._decoder: dict[int, str] = {i: c for (i, c) in enumerate(self._chars)}
    
    def encode(self, text: str) -> list[int]:
        return [self._encoder[c] for c in text]
    
    def decode(self, encoded_text: list[int]) -> str:
        return ''.join([self._decoder[i] for i in encoded_text])
    
    def __len__(self):
        return len(self._chars)
    
    def __contains__(self, character: str):
        if len(character)!= 1:
            raise ValueError('Character must be a single character.')
        return character in self._encoder
