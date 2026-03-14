import argparse
import json
from pathlib import Path

import numpy as np

def load_artifacts(emb_path: Path, vocab_path: Path) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Load embeddings and vocabulary artifacts from disk."""
    embeddings = np.load(emb_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx2word = vocab["idx2word"]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return embeddings, idx2word, word2idx

def normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row in a 2D array, leaving zero rows unchanged."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def most_similar(
    word: str,
    embeddings: np.ndarray,
    idx2word: list[str],
    word2idx: dict[str, int],
    k: int,
) -> list[tuple[str, float]]:
    """Return the top-k most similar words to a query word by cosine similarity."""
    if word not in word2idx:
        return []
    w_idx = word2idx[word]
    vecs = normalize_rows(embeddings.astype(np.float32, copy=False))
    query = vecs[w_idx]
    scores = vecs @ query
    scores[w_idx] = -1.0
    top_idx = np.argpartition(-scores, range(k))[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(idx2word[i], float(scores[i])) for i in top_idx]

def analogy(
    a: str,
    b: str,
    c: str,
    embeddings: np.ndarray,
    idx2word: list[str],
    word2idx: dict[str, int],
    k: int,
) -> list[tuple[str, float]]:
    """Solve analogies of the form A - B + C and return top-k candidates."""
    for w in (a, b, c):
        if w not in word2idx:
            return []
    vecs = normalize_rows(embeddings.astype(np.float32, copy=False))
    # A - B + C
    query = vecs[word2idx[a]] - vecs[word2idx[b]] + vecs[word2idx[c]]
    query = query / (np.linalg.norm(query) + 1e-12)
    scores = vecs @ query
    for w in (a, b, c):
        scores[word2idx[w]] = -1.0
    top_idx = np.argpartition(-scores, range(k))[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(idx2word[i], float(scores[i])) for i in top_idx]

def main() -> None:
    """Parse CLI arguments and run similarity or analogy queries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="embeddings.npy")
    parser.add_argument("--vocab", default="vocab.json")
    parser.add_argument("--word", default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--analogy", nargs=3, default=None, metavar=("A", "B", "C"))
    args = parser.parse_args()

    embeddings, idx2word, word2idx = load_artifacts(Path(args.emb), Path(args.vocab))

    if args.word:
        sims = most_similar(args.word, embeddings, idx2word, word2idx, args.topk)
        if not sims:
            print(f"Word not in vocab: {args.word}")
        else:
            for w, s in sims:
                print(f"{w}\t{s:.4f}")

    if args.analogy:
        a, b, c = args.analogy
        sims = analogy(a, b, c, embeddings, idx2word, word2idx, args.topk)
        if not sims:
            print(f"Analogy words not in vocab: {a}, {b}, {c}")
        else:
            for w, s in sims:
                print(f"{w}\t{s:.4f}")

    if not args.word and not args.analogy:
        print(f"Loaded embeddings: {embeddings.shape[0]} x {embeddings.shape[1]}")
        print(f"Vocab size: {len(idx2word)}")
        print("Use --word or --analogy to test.")

if __name__ == "__main__":
    main()
