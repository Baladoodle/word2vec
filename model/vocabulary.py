from dataclasses import dataclass, field
from collections import Counter
from config import Config

@dataclass
class Vocabulary:
    """A vocabulary for tokenizing text."""
    min_count: int = 5
    max_size: int | None = None

    word2idx: dict[str, int] = field(default_factory=dict)
    idx2word: list[str]      = field(default_factory=list)
    counts:   Counter        = field(default_factory=Counter)

    def build(self, tokens: list[str]) -> None:
        self.counts = Counter(tokens)
        kept = [(w, c) for w, c in self.counts.items() if c >= self.min_count]
        kept.sort(key=lambda x: (-x[1], x[0]))
        if self.max_size is not None:
            kept = kept[: self.max_size]
        kept = [w for w, _ in kept]
        self.idx2word = ["<UNK>"] + kept # <UNK> is reserved for unknown words
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def encode(self, token: str) -> int:
        return self.word2idx.get(token, 0)

    def __len__(self) -> int:
        return len(self.idx2word)
    
    def __str__(self) -> str:
        items = [(w, c) for w, c in self.counts.items() if c >= self.min_count]
        items.sort(key=lambda x: (-x[1], x[0]))
        shown = items[: Config.vocab_print]
        lines = [f"{w}: {c}" for w, c in shown]
        remaining = len(items) - len(shown)
        if remaining > 0:
            lines.append(f"... + {remaining} more")
        return "\n".join(lines)
    
    def lookup_token(self, idx: int) -> str:
        return self.idx2word[idx]
    
    def lookup_index(self, token: str) -> int:
        return self.word2idx.get(token, 0)
