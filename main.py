import logging
import os

from bank import Bank


bank_id = "test_bank"


def main():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logging.info("Reading bank '{}'".format(bank_id))
    bank = Bank(bank_id)
    

if __name__ == "__main__":
    main()