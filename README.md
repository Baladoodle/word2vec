# word2vec

Based on the word2vec paper by Mikolov et al. (2013).

This repo is a pure NumPy implementation of word2vec (skip-gram with negative sampling). It includes the full optimization procedure (forward pass, loss, gradients, and parameter updates) and is made only using NumPy.

**Features**
- Skip-gram with negative sampling (SGNS)
- Full training loop in NumPy 
- Word2vec-style subsampling
- Alias-table unigram sampling with exponent

**Requirements**
- Python 3.10+
- `pip install -r requirements.txt`
- CUDA 13.0+ required for CuPy (optional)

**Quickstart**
```bash
python main.py
```

**Test Script**:
Test the output embeddings in `extras/test.py`:
```bash
python extras/test.py
```
Parameters:
- `--emb`: Path to embeddings file (default: `embeddings.npy`)
- `--vocab`: Path to vocab file (default: `vocab.json`)
- `--word`: Query word for nearest neighbors (default: none)
- `--topk`: Number of results to return (default: `10`)
- `--analogy`: Three words `A B C` for analogy `A - B + C` (default: none)

**Configuration**:
Edit `config.py` to adjust the dataset, model size, training steps, and output paths. Any suitable text dataset is supported; the default uses a Hugging Face dataset.

**Outputs**
- `embeddings.npy`: input embedding matrix (NumPy)
- `vocab.json`: `idx2word` list and token counts
