from datasets import load_dataset
from data.corpus import token_stream
from model.vocabulary import Vocabulary
from config import Config

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

vocab = Vocabulary(min_count=Config.min_count, max_size=Config.max_vocab)
vocab.build(token_stream(ds["text"]))
print(vocab)

print(vocab.lookup_index("the"))
print(vocab.lookup_token(10000))
