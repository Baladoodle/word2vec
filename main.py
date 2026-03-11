from datasets import load_dataset
from data.corpus import token_stream
from model.training import (
    init_embeddings,
    iter_negative_sampling_batches,
    negative_sampling_loss,
)
from model.vocabulary import Vocabulary
from config import Config

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

tokens = token_stream(ds["text"])
if Config.train_tokens_limit is not None:
    tokens = tokens[: Config.train_tokens_limit]

vocab = Vocabulary(min_count=Config.min_count, max_size=Config.max_vocab)
vocab.build(tokens)
print(vocab)

token_ids = [vocab.encode(t) for t in tokens]
w_in, w_out = init_embeddings(len(vocab), Config.embedding_dim, seed=Config.seed)

for centers, contexts, negatives in iter_negative_sampling_batches(
    token_ids,
    vocab,
    window_size=Config.window_size,
    batch_size=Config.batch_size,
    negatives=Config.negatives,
    seed=Config.seed,
):
    loss = negative_sampling_loss(w_in, w_out, centers, contexts, negatives)
    print(centers.shape, contexts.shape, negatives.shape, loss)
    break

print(vocab.lookup_index("the"))
print(vocab.lookup_token(10000)) # On pure coincidence the word is "kanye"
