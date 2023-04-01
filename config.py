"""Configuration file with global variables."""

# SAMPLE PREPROCESSING

SAMPLE_RATE = 16000 # Hz
SAMPLE_DUR = 0.5 # Seconds
SAMPLE_LEN = int(SAMPLE_DUR*SAMPLE_RATE)

# DATASET

BUFFER_SIZE = 32
DATA_PATH = "data"
SAMPLE_DATA_FILE = "samples.csv"
OUT_BANK_PREFIX = "_"


# MODEL

TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
VAL_SIZE = 0.1

