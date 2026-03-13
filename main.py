from datasets import load_dataset
import numpy as np
from data.corpus import token_stream, subsample_token_ids
from model.training import (
    init_embeddings,
    iter_negative_sampling_batches,
    negative_sampling_backward_update,
)
from model.backend import to_numpy
from model.vocabulary import Vocabulary
from config import Config
import json

def load_tokens():
    """Load and tokenize the training dataset into a flat token list."""
    ds = load_dataset(
        Config.dataset_name,
        Config.dataset_config,
        split=Config.dataset_split,
    )
    tokens = token_stream(ds["text"])
    if Config.train_tokens_limit is not None:
        tokens = tokens[: Config.train_tokens_limit]
    return tokens

def build_vocab(tokens: list[str]) -> Vocabulary:
    """Build and return a Vocabulary from tokens using Config settings."""
    vocab = Vocabulary(min_count=Config.min_count, max_size=Config.max_vocab)
    vocab.build(tokens)
    return vocab

def maybe_subsample(token_ids: list[int], vocab: Vocabulary) -> list[int]:
    """Optionally apply word2vec-style subsampling based on Config."""
    if Config.subsample_t is None or Config.subsample_t <= 0:
        return token_ids
    counts_by_id = [0] * len(vocab)
    for idx in token_ids:
        counts_by_id[idx] += 1
    return subsample_token_ids(
        token_ids,
        counts_by_id,
        Config.subsample_t,
        seed=Config.seed,
    )

def train_skipgram(
    token_ids: list[int],
    vocab: Vocabulary,
) -> tuple[np.ndarray, np.ndarray]:
    """Train skip-gram with negative sampling and return input/output embeddings."""
    w_in, w_out = init_embeddings(len(vocab), Config.embedding_dim, seed=Config.seed)

    total_steps = max(1, Config.max_steps)
    global_step = 0

    for epoch in range(Config.epochs):
        for centers, contexts, negatives in iter_negative_sampling_batches(
            token_ids,
            vocab,
            window_size=Config.window_size,
            batch_size=Config.batch_size,
            negatives=Config.negatives,
            seed=None if Config.seed is None else Config.seed + epoch,
        ):
            t = min(global_step / max(1, total_steps - 1), 1.0)
            lr = Config.lr_start + (Config.lr_end - Config.lr_start) * t
            loss = negative_sampling_backward_update(
                w_in, w_out, centers, contexts, negatives, lr=lr
            )
            if global_step % Config.loss_log_every == 0:
                print(f"epoch={epoch+1} step={global_step} lr={lr:.6f} loss={loss:.4f}")
            global_step += 1
            if global_step >= Config.max_steps:
                return w_in, w_out

    return w_in, w_out

def save_outputs(vocab: Vocabulary, w_in: np.ndarray) -> None:
    """Persist vocab and embeddings to disk based on Config paths."""
    with open(Config.vocab_out, "w", encoding="utf-8") as f:
        json.dump(
            {"idx2word": vocab.idx2word, "counts": dict(vocab.counts)},
            f,
            ensure_ascii=False,
        )
    np.save(Config.embeddings_out, to_numpy(w_in))

def main():
    tokens = load_tokens()                          # Load tokens from disk
    vocab = build_vocab(tokens)                     # Build vocabulary from tokens

    token_ids = [vocab.encode(t) for t in tokens]   # Encode tokens to token IDs
    token_ids = maybe_subsample(token_ids, vocab)
    w_in, _ = train_skipgram(token_ids, vocab)      # Train skip-gram model

    save_outputs(vocab, w_in)                       # Output vocabulary and embeddings to disk

if __name__ == "__main__":
    main()
