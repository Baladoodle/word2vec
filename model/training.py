import numpy as np
from data.corpus import iterate_center_context, batch_pairs
from model.sampling import build_alias_table, sample_alias
from model.vocabulary import Vocabulary

def init_embeddings(
    vocab_size: int,
    embedding_dim: int,
    seed: int | None = None,
    scale: float = 0.01,
):
    """Initialize input/output embeddings with small random values."""
    rng = np.random.default_rng(seed)
    w_in = rng.normal(0.0, scale, size=(vocab_size, embedding_dim)).astype(np.float32)
    w_out = rng.normal(0.0, scale, size=(vocab_size, embedding_dim)).astype(np.float32)
    return w_in, w_out


def negative_sampling_loss(
    w_in: np.ndarray,
    w_out: np.ndarray,
    centers: np.ndarray,
    contexts: np.ndarray,
    negatives: np.ndarray,
) -> float:
    """Compute mean negative-sampling loss for a batch."""
    center_vecs = w_in[centers]
    pos_vecs = w_out[contexts]
    dot_pos = np.sum(center_vecs * pos_vecs, axis=1)

    neg_vecs = w_out[negatives]
    dot_neg = np.sum(center_vecs[:, None, :] * neg_vecs, axis=2)

    log_sigmoid_pos = -np.logaddexp(0.0, -dot_pos)
    log_sigmoid_neg = -np.logaddexp(0.0, dot_neg)
    loss = -(log_sigmoid_pos + np.sum(log_sigmoid_neg, axis=1))
    return float(np.mean(loss))

def iter_negative_sampling_batches(
    token_ids: list[int],
    vocab: Vocabulary,
    window_size: int,
    batch_size: int,
    negatives: int,
    seed: int | None = None,
):
    """Yield (centers, contexts, negatives) batches for negative sampling."""
    prob, alias = build_alias_table(vocab)
    pair_iter = iterate_center_context(
        token_ids,
        window_size=window_size,
        seed=seed,
    )

    for batch_idx, (centers, contexts) in enumerate(batch_pairs(pair_iter, batch_size)):
        batch_seed = None if seed is None else seed + batch_idx
        neg = sample_alias(
            prob,
            alias,
            size=len(centers) * negatives,
            seed=batch_seed,
        ).reshape(len(centers), negatives)
        yield centers, contexts, neg
