from model.backend import xp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
from model.vocabulary import Vocabulary
from config import Config

# Alias table: https://bfraboni.github.io/data/alias2022/alias-table.pdf
def build_alias_table(vocab: Vocabulary) -> tuple["np.ndarray", "np.ndarray"]:
    """Build alias tables for O(1) sampling from unigram^exp distribution."""

    freq = xp.array([vocab.counts.get(w, 0) ** Config.freq_exponent for w in vocab.idx2word], dtype=xp.float64)
    freq[0] = 0  # <UNK>
    total = freq.sum()
    if total <= 0:
        n = len(vocab)
        return xp.zeros(n, dtype=xp.float64), xp.zeros(n, dtype=xp.int64)

    probs = freq / total
    n = len(probs)
    scaled = probs * n

    alias = xp.zeros(n, dtype=xp.int64)
    prob = xp.zeros(n, dtype=xp.float64)

    small = [i for i, p in enumerate(scaled.tolist()) if p < 1.0]
    large = [i for i, p in enumerate(scaled.tolist()) if p >= 1.0]

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] - (1.0 - scaled[s])
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    # Remaining entries have prob 1
    for i in small + large:
        prob[i] = 1.0
        alias[i] = i

    return prob, alias

def sample_alias(
    prob: "np.ndarray",
    alias: "np.ndarray",
    size: int,
    seed: int | None = Config.unigram_seed,
) -> "np.ndarray":
    """Sample indices using alias tables in O(1) per sample."""
    n = len(prob)
    if n == 0:
        return xp.zeros(size, dtype=xp.int64)
    rng = xp.random.default_rng(seed)
    idx = rng.integers(0, n, size=size)
    coin = rng.random(size)
    return xp.where(coin < prob[idx], idx, alias[idx]).astype(xp.int64)
