import logging
import os
import datetime as dt

from config import *
from src.bank import Bank
from src.models.autoencoder import Autoencoder
from src.models.variationalautoencoder import VariationalAutoencoder

BANK_PATH = "data/sample_pack"


def main():
    """Main workflow of the program.
        - Reads bank passed as bank_id
        """
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logging.info("Reading bank '{}'".format(BANK_PATH))
    bank = Bank(BANK_PATH)
    autoencoder = VariationalAutoencoder(
        input_shape = AE_INPUT_SHAPE,
        filters = AE_CONV_FILTERS,
        kernels = AE_CONV_KERNELS,
        strides = AE_CONV_STRIDES,
        latent_space_shape = AE_LATENT_SPACE_SHAPE
        )
    
    x_train, y_train, x_test, y_test, x_val, y_val = bank.load_spectograms()

    autoencoder.compile(AE_LEARNING_RATE)
    autoencoder.train(x_train, AE_BATCH_SIZE, AE_EPOCHS)


def init_logs():
    """Inits logging."""
    if not os.path.exists("log/"):
        os.mkdir("log/")
    if not os.path.exists(f"log/{dt.datetime.now().strftime('%Y%m%d')}/"):
        os.mkdir(f"log/{dt.datetime.now().strftime('%Y%m%d')}/")
    
    # Log to file
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s] %(levelname)s: %(module)s:%(funcName)s: %(message)s \t",
                        filename=f"log/{dt.datetime.now().strftime('%Y%m%d')}/debug_{dt.datetime.now().strftime('%H%M%S')}.txt")
    
    # Log to screen
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(module)s:%(funcName)s: %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    

if __name__ == "__main__":
    init_logs()

    logging.info("Initializing main...")
    try:
        main()
    except Exception as e:
        logging.critical(f'Catched exception {e}')
        raise e