import re # Regular expressions
import random

def tokenize(text: str, keep_punct: bool = True) -> list[str]:
    """Tokenize a string. Optionally keep punctuation; always lowercase."""

    text = text.lower()
    if keep_punct:
        return text.split()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def token_stream(lines: list[str], keep_punct: bool = True) -> list[str]:
    """Build a token stream from a list of strings."""

    tokens: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens.extend(tokenize(line, keep_punct=keep_punct))
    return tokens



def iterate_center_context(
    token_ids: list[int],
    window_size: int,
    seed: int | None = None,
):
    """Yield (center_id, context_id) with a dynamic window."""
    rng = random.Random(seed)
    n = len(token_ids)
    for i in range(n):
        window = rng.randint(1, window_size)
        left = max(0, i - window)
        right = min(n, i + window + 1)
        center_id = token_ids[i]
        for j in range(left, right):
            if j == i:
                continue
            yield center_id, token_ids[j]


def batch_pairs(pairs_iter, batch_size: int):
    """Batch (center_id, context_id) pairs into numpy arrays."""
    import numpy as np

    centers: list[int] = []
    contexts: list[int] = []
    for center_id, context_id in pairs_iter:
        centers.append(center_id)
        contexts.append(context_id)
        if len(centers) >= batch_size:
            yield np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)
            centers.clear()
            contexts.clear()
    if centers:
        yield np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)
