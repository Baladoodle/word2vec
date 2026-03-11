import numpy as np
from data.corpus import iterate_center_context, batch_pairs
from model.sampling import build_alias_table, sample_alias
from model.vocabulary import Vocabulary

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
