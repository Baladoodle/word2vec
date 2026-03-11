from datasets import load_dataset
from data.corpus import tokenize
from model.vocabulary import Vocabulary
from config import Config

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

full_text = " ".join(ds["text"])

print(full_text[:1000])

vocab = Vocabulary(min_count=Config.min_count)
vocab.build(tokenize(full_text))
print(vocab)

print(vocab.lookup_index("the"))
print(vocab.lookup_token(10000))