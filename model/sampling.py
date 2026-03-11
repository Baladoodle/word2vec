import numpy as np
from model.vocabulary import Vocabulary
from config import Config

def build_unigram(
    vocab: Vocabulary,
    table_size: int = Config.table_size,
    seed: int | None = Config.unigram_seed,
) -> np.ndarray: # one million

    freq = np.array([vocab.counts.get(w, 0) ** Config.freq_exponent for w in vocab.idx2word], dtype=np.float64)
    freq[0] = 0             # because of <UNK> token
    freq_sum = freq.sum()
    if freq_sum <= 0:
        return np.zeros(table_size, dtype=np.int64)
    freq /= freq_sum        # normalize

    rng = np.random.default_rng(seed)
    return rng.choice(len(vocab), table_size, p=freq).astype(np.int64)


def build_alias_table(vocab: Vocabulary) -> tuple[np.ndarray, np.ndarray]:
    """Build alias tables for O(1) sampling from unigram^exp distribution."""
    freq = np.array([vocab.counts.get(w, 0) ** Config.freq_exponent for w in vocab.idx2word], dtype=np.float64)
    freq[0] = 0  # <UNK>
    total = freq.sum()
    if total <= 0:
        n = len(vocab)
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.int64)

    probs = freq / total
    n = len(probs)
    scaled = probs * n

    alias = np.zeros(n, dtype=np.int64)
    prob = np.zeros(n, dtype=np.float64)

    small = [i for i, p in enumerate(scaled) if p < 1.0]
    large = [i for i, p in enumerate(scaled) if p >= 1.0]

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
    prob: np.ndarray,
    alias: np.ndarray,
    size: int,
    seed: int | None = Config.unigram_seed,
) -> np.ndarray:
    """Sample indices using alias tables in O(1) per sample."""
    n = len(prob)
    if n == 0:
        return np.zeros(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=size)
    coin = rng.random(size)
    return np.where(coin < prob[idx], idx, alias[idx]).astype(np.int64)
