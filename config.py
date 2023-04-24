"""Configuration file with global variables."""

# BANK
BANK_ID = "test_bank_big"

SAMPLE_RATE = 16000 # Hz
SAMPLE_DUR = 1 # Seconds
SAMPLE_LEN = int(SAMPLE_DUR*SAMPLE_RATE)
OUT_BANK_SUFFIX = "_"

# DATASET
BUFFER_SIZE = 32
DATA_PATH = "data"
SAMPLE_DATA_FILE = "samples.csv"
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# FFT
FRAME_LENGTH = 255
FRAME_STEP=128

# AUTOENCODER
AE_INPUT_SHAPE = (28, 28, 1)
AE_CONV_FILTERS = (32, 64, 64, 64)
AE_CONV_KERNELS = (3, 3, 3, 3)
AE_CONV_STRIDES =  (1, 2, 2, 1)
AE_LATENT_SPACE_SHAPE = 8
AE_LEARNING_RATE = 0.01
AE_BATCH_SIZE = 128
AE_EPOCHS = 10
AE_LOSS_WEIGHT = 10000


