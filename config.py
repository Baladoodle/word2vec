from dataclasses import dataclass

@dataclass
class Config:
    min_count:      int         = 5
    freq_exponent:  float       = 0.75          # keep it smaller than 1 unless you want very random words

    table_size:     int         = 1000000