"""Configuration file with global variables."""

# SAMPLE PREPROCESSING

SAMPLE_RATE = 16000 # Hz
SAMPLE_DUR = 1 # Seconds
SAMPLE_LEN = int(SAMPLE_DUR*SAMPLE_RATE)

# DATASET

BUFFER_SIZE = 32
LABELS = ['hardness', 'depth', 'brightness', 'roughness', 'warmth', 'sharpness', 'boominess']
DATA_PATH = "data"
SAMPLE_DATA_FILE = "samples.csv"
TIMBRAL_DATA_FILE = "timbre.csv"
OUT_BANK_PREFIX = "_"


# MODEL

TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
VAL_SIZE = 0.1

