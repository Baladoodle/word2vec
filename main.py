from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

full_text = " ".join(ds["text"])

print(full_text[:10000])