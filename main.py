from datasets import load_dataset
from data.corpus import token_stream, iterate_center_context
from model.vocabulary import Vocabulary
from config import Config

# ids = [0, 1, 2, 3]
# pairs = list(iterate_center_context(ids, window_size=2, seed=0))
# print(pairs)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

tokens = token_stream(ds["text"])
if Config.train_tokens_limit is not None:
    tokens = tokens[: Config.train_tokens_limit]

vocab = Vocabulary(min_count=Config.min_count, max_size=Config.max_vocab)
vocab.build(tokens)
print(vocab)

token_ids = [vocab.encode(t) for t in tokens]

print(vocab.lookup_index("the"))
print(vocab.lookup_token(10000)) # On pure coincidence the word is "kanye"
