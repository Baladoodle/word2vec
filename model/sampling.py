import numpy as np
from model.vocabulary import Vocabulary
from config import Config

def build_unigram(vocab: Vocabulary, table_size: int = Config.table_size) -> np.ndarray: # one million

    freq = np.array([vocab.counts.get(w, 0) ** Config.freq_exponent for w in vocab.idx2word])
    freq[0] = 0             # because of <UNK> token
    freq /= freq.sum()      # normalize
    return np.random.choice(len(vocab), table_size, p=freq)


