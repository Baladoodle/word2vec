from model.backend import xp, rng
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
from data.corpus import iterate_center_context, batch_pairs
from model.sampling import build_alias_table, sample_alias
from model.vocabulary import Vocabulary

def init_embeddings(
    vocab_size: int,
    embedding_dim: int,
    seed: int | None = None,
    scale: float = 0.01,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Initialize input/output embeddings with small random values."""
    gen = rng(seed)
    w_in = gen.normal(0.0, scale, size=(vocab_size, embedding_dim)).astype(xp.float32)
    w_out = gen.normal(0.0, scale, size=(vocab_size, embedding_dim)).astype(xp.float32)
    return w_in, w_out


def negative_sampling_loss(
    w_in: "np.ndarray",
    w_out: "np.ndarray",
    centers: "np.ndarray",
    contexts: "np.ndarray",
    negatives: "np.ndarray",
) -> float:
    """Compute mean negative-sampling loss for a batch."""
    center_vecs = w_in[centers]
    pos_vecs = w_out[contexts]
    dot_pos = xp.sum(center_vecs * pos_vecs, axis=1)

    neg_vecs = w_out[negatives]
    dot_neg = xp.sum(center_vecs[:, None, :] * neg_vecs, axis=2)

    log_sigmoid_pos = -xp.logaddexp(0.0, -dot_pos)
    log_sigmoid_neg = -xp.logaddexp(0.0, dot_neg)
    loss = -(log_sigmoid_pos + xp.sum(log_sigmoid_neg, axis=1))
    return float(xp.mean(loss))

def negative_sampling_backward_update(
    w_in: "np.ndarray",
    w_out: "np.ndarray",
    centers: "np.ndarray",
    contexts: "np.ndarray",
    negatives: "np.ndarray",
    lr: float,
) -> float:
    """
    Skip-gram negative sampling with forwards and backwards update.
    """
    # Forward
    center_vecs = w_in[centers]                       # (B, D)
    pos_vecs = w_out[contexts]                        # (B, D)
    neg_vecs = w_out[negatives]                       # (B, K, D)

    dot_pos = xp.sum(center_vecs * pos_vecs, axis=1)  # (B,)
    dot_neg = xp.sum(center_vecs[:, None, :] * neg_vecs, axis=2)  # (B, K)

    # Sigmoid
    sig_pos = 1.0 / (1.0 + xp.exp(-dot_pos))          # (B,)
    sig_neg = 1.0 / (1.0 + xp.exp(-dot_neg))          # (B, K)

    # Loss 
    log_sigmoid_pos = -xp.logaddexp(0.0, -dot_pos)
    log_sigmoid_neg = -xp.logaddexp(0.0, dot_neg)
    loss = -(log_sigmoid_pos + xp.sum(log_sigmoid_neg, axis=1))
    mean_loss = float(xp.mean(loss))

    # Gradients
    # We want sig_pos to be near 1 and sig_neg to be near 0. Smaller g_pos and g_neg means less changes to w_in and w_out
    # These act as error signals
    g_pos = (sig_pos - 1.0).astype(xp.float32)        # (B,)
    g_neg = sig_neg.astype(xp.float32)                # (B, K)

    # Note: [:, None] just specifies which dimensions to multiply with
    # If not used, NumPy doesn't know how to multiply (B,) with (B, D), etc.

    # Simpler than it looks: Positive gradients * Positive vectors + Sum(Negative gradients * Negative vectors)
    grad_w_in = g_pos[:, None] * pos_vecs + xp.sum(g_neg[:, :, None] * neg_vecs, axis=1)    #(B, D)                                         # (B, D)

    # Positive gradients * Center vectors
    grad_w_out_pos = g_pos[:, None] * center_vecs     # (B, D)

    # Negative gradients * Center vectors
    grad_w_out_neg = g_neg[:, :, None] * center_vecs[:, None, :]  # (B, K, D)

    xp.add.at(w_in, centers, -lr * grad_w_in)
    xp.add.at(w_out, contexts, -lr * grad_w_out_pos)

    # ravel() - converts to 1D array
    # reshape() - converts to 2D array
    # This is done to match the shape of w_out
    xp.add.at(w_out, negatives.ravel(), -lr * grad_w_out_neg.reshape(-1, w_out.shape[1]))

    return mean_loss

def iter_negative_sampling_batches(
    token_ids: list[int],
    vocab: Vocabulary,
    window_size: int,
    batch_size: int,
    negatives: int,
    seed: int | None = None,
    prob: "np.ndarray | None" = None,
    alias: "np.ndarray | None" = None,
):
    """Yield (centers, contexts, negatives) batches for negative sampling."""
    if prob is None or alias is None:
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
