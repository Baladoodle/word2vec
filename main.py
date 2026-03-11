from datasets import load_dataset
from data.corpus import tokenize
from model.vocabulary import Vocabulary

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

full_text = " ".join(ds["text"])

print(full_text[:1000])

vocab = Vocabulary(min_count=5)
vocab.build(tokenize(full_text))
print(vocab)