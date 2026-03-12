from dataclasses import dataclass

@dataclass
class Config:
    min_count:          int         = 5             # Remove words with less than min_count occurences.
    freq_exponent:      float       = 0.75          # Frequency exponent
    unigram_seed:       int | None  = 1337          # Negative sampling seed
    max_vocab:          int | None  = 50000         # Max vocabulary size
    vocab_print:        int         = 100           # Vocab __str__ limit
    table_size:         int         = 1000000       # Unigram table size (Deprecated)

    embedding_dim:      int         = 128               # Embedding width
    window_size:        int         = 5                 # Context window
    negatives:          int         = 5                 # Negatives per positive
    batch_size:         int         = 1024              # Training batch size
    epochs:             int         = 1                 # Training epochs
    lr_start:           float       = 0.05              # Initial learning rate
    lr_end:             float       = 0.005             # Final learning rate
    train_tokens_limit: int | None  = 10_000_000        # Cap train tokens
    max_steps:          int         = 60000             # Hard cap on training steps
    seed:               int | None  = 1337              # Training seed

    embeddings_out:     str         = "embeddings.npy"          # Embeddings output path
    vocab_out:          str         = "vocab.json"              # Vocab output path
    loss_log_every:     int         = 1000                      # Log loss interval
    use_cupy:           bool        = False                     # Uses CuPy instead of NumPy for GPU accel. Requires CUDA 13.0+
    dataset_name:       str         = "Salesforce/wikitext"     # HF dataset name
    dataset_config:     str         = "wikitext-103-raw-v1"     # HF dataset config
    dataset_split:      str         = "train"                   # HF dataset split
