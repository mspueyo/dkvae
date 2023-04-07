"""Configuration file with global variables."""

# BANK
BANK_ID = "test_bank_big"

SAMPLE_RATE = 16000 # Hz
SAMPLE_DUR = 0.5 # Seconds
SAMPLE_LEN = int(SAMPLE_DUR*SAMPLE_RATE)
OUT_BANK_SUFFIX = "_"

# DATASET
BUFFER_SIZE = 32
DATA_PATH = "data"
SAMPLE_DATA_FILE = "samples.csv"


# MODEL
TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
VAL_SIZE = 0.1

