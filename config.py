"""Configuration file with global variables."""

SAMPLE_RATE = 22050 # Hz
SAMPLE_DUR = 0.74 # Seconds
SAMPLE_LEN = int(SAMPLE_DUR*SAMPLE_RATE)
OUT_BANK_SUFFIX = "_"

# DATASET
BUFFER_SIZE = 32
SAMPLE_DATA_FILE = "samples.csv"
TEST_SIZE = 0.1
VAL_SIZE = 0.05

# FFT
FFT_FRAME_SIZE = 512
FFT_HOP_LENGTH = 256

# AUTOENCODER
AE_INPUT_SHAPE = (512, 128, 1)
AE_CONV_FILTERS = (1024, 512, 256, 128, 64)
AE_CONV_KERNELS = (3, 3, 3, 3, 3)
AE_CONV_STRIDES =  (2, 2, 2, (2, 1))
AE_LATENT_SPACE_SHAPE = 8
AE_LEARNING_RATE = 0.00000005
AE_BATCH_SIZE = 128
AE_EPOCHS = 150
AE_LOSS_WEIGHT = 0.000001


