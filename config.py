from dataclasses import dataclass

@dataclass
class Config:
    min_count:      int         = 5
    freq_exponent:  float       = 0.75          # keep it smaller than 1 unless you want very random words
    max_vocab:      int | None  = 50000         # set None to disable cap
    unigram_seed:  int | None  = 1337          # set None for non-deterministic sampling

    table_size:     int         = 1000000
