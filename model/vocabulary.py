from dataclasses import dataclass, field
from collections import Counter

@dataclass
class Vocabulary:
    """A vocabulary for tokenizing text."""
    min_count: int = 5

    word2idx: dict[str, int] = field(default_factory=dict)
    idx2word: list[str]      = field(default_factory=list)
    counts:   Counter        = field(default_factory=Counter)

    def build(self, tokens: list[str]) -> None:
        self.counts = Counter(tokens)
        kept = [w for w, c in self.counts.items() if c >= self.min_count]
        kept.sort()
        self.idx2word = ["<unknown>"] + kept # <unknown> is reserved for unknown words
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def encode(self, token: str) -> int:
        return self.word2idx.get(token, 0)

    def __len__(self) -> int:
        return len(self.idx2word)
    
    def __str__(self) -> str:
        return str(self.counts)