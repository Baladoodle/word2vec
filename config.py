from dataclasses import dataclass

@dataclass
class Config:
    min_count:          int             = 5             # Remove words with less than min_count occurences.
    freq_exponent:      float           = 0.75          # Frequency exponent
    unigram_seed:       int | None      = 1337          # Negative sampling seed
    vocab_print:        int             = 100           # Vocab __str__ limit
    table_size:         int             = 1000000       # Unigram table size (Deprecated)

    # # GPU (3056.2s on a 4060 laptop)
    # embedding_dim:      int             = 256           # Embedding width
    # window_size:        int             = 5             # Context window
    # negatives:          int             = 8             # Negatives per positive
    # batch_size:         int             = 1024          # Training batch size
    # epochs:             int             = 10            # Training epochs
    # lr_start:           float           = 0.025         # Initial learning rate
    # lr_end:             float           = 0.0001        # Final learning rate
    # train_tokens_limit: int | None      = 50_000_000    # Cap train tokens
    # max_steps:          int | None      = None          # Hard cap on training steps
    # max_vocab:          int | None      = 50000         # Max vocabulary size
    # seed:               int | None      = 1337          # Training seed
    # subsample_t:        float | None    = 0.001         # word2vec subsampling threshold
    # use_cupy:           bool            = True          # Uses CuPy instead of NumPy for GPU accel. Requires CUDA 13.0+

    # CPU (4223.0s on a Ryzen 7 7435HS)
    embedding_dim:      int             = 128           # Embedding width
    window_size:        int             = 5             # Context window
    negatives:          int             = 5             # Negatives per positive
    batch_size:         int             = 512           # Training batch size
    epochs:             int             = 5             # Training epochs
    lr_start:           float           = 0.025         # Initial learning rate
    lr_end:             float           = 0.0001        # Final learning rate
    train_tokens_limit: int | None      = 25_000_000    # Cap train tokens
    max_steps:          int | None      = None          # Hard cap on training steps
    max_vocab:          int | None      = 30000         # Max vocabulary size
    seed:               int | None      = 1337          # Training seed
    subsample_t:        float | None    = 0.001         # word2vec subsampling threshold
    use_cupy:           bool            = False         # Uses CuPy instead of NumPy for GPU accel. Requires CUDA 13.0+

    embeddings_out:     str             = "embeddings_cpu.npy"          # Embeddings output path
    vocab_out:          str             = "vocab_cpu.json"              # Vocab output path
    loss_log_every:     int             = 1000                          # Log loss interval
    dataset_name:       str             = "Salesforce/wikitext"         # HF dataset name
    dataset_config:     str             = "wikitext-103-raw-v1"         # HF dataset config
    dataset_split:      str             = "train"                       # HF dataset split
