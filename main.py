from datasets import load_dataset
import math
import numpy as np
import time
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

def subsample(token_ids: list[int], vocab: Vocabulary) -> list[int]:
    """Randomly remove tokens from token_ids."""
    counts_by_id = [vocab.counts.get(word, 0) for word in vocab.idx2word]
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
    from model.sampling import build_alias_table
    prob, alias = build_alias_table(vocab)

    total_steps = max(1, Config.max_steps) if Config.max_steps is not None else 1
    start_time = time.time()
    steps_per_epoch_est = None
    if Config.max_steps is None:
        # Expected contexts per token ≈ (window_size + 1) with uniform dynamic window.
        expected_pairs = len(token_ids) * (Config.window_size + 1)
        steps_per_epoch_est = max(1, math.ceil(expected_pairs / Config.batch_size))
    global_step = 0

    for epoch in range(Config.epochs):
        step_in_epoch = 0
        for centers, contexts, negatives in iter_negative_sampling_batches(
            token_ids,
            vocab,
            window_size=Config.window_size,
            batch_size=Config.batch_size,
            negatives=Config.negatives,
            seed=None if Config.seed is None else Config.seed + epoch,
            prob=prob,
            alias=alias,
        ):
            if Config.max_steps is None and steps_per_epoch_est is not None:
                denom = max(1, Config.epochs * steps_per_epoch_est - 1)
                t = min((epoch * steps_per_epoch_est + step_in_epoch) / denom, 1.0)
            else:
                t = min(global_step / max(1, total_steps - 1), 1.0)
            lr = Config.lr_start + (Config.lr_end - Config.lr_start) * t
            loss = negative_sampling_backward_update(
                w_in, w_out, centers, contexts, negatives, lr=lr
            )
            if global_step % Config.loss_log_every == 0:
                elapsed_s = time.time() - start_time
                print(
                    f"epoch={epoch+1} step={global_step} lr={lr:.6f} "
                    f"loss={loss:.4f} elapsed_s={elapsed_s:.1f}"
                )
            global_step += 1
            step_in_epoch += 1
            if Config.max_steps is not None and global_step >= Config.max_steps:
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
    token_ids = [idx for idx in token_ids if idx != 0]  # Drop <UNK> from training
    if Config.subsample_t > 0: token_ids = subsample(token_ids, vocab)  # Subsample tokens
    w_in, _ = train_skipgram(token_ids, vocab)      # Train skip-gram model

    save_outputs(vocab, w_in)                       # Output vocabulary and embeddings to disk

if __name__ == "__main__":
    main()
