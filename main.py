from datasets import load_dataset
from data.corpus import token_stream
from model.training import iter_negative_sampling_batches
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

for centers, contexts, negatives in iter_negative_sampling_batches(
    token_ids,
    vocab,
    window_size=Config.window_size,
    batch_size=Config.batch_size,
    negatives=Config.negatives,
    seed=Config.seed,
):
    print(centers.shape, contexts.shape, negatives.shape)
    break

print(vocab.lookup_index("the"))
print(vocab.lookup_token(10000)) # On pure coincidence the word is "kanye"
