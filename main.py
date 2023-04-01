import logging
import os
import datetime as dt

from bin.bank import Bank


bank_id = "test_bank_big"


def main():
    """Main workflow of the program.
        - Reads bank passed as bank_id
        """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logging.info("Reading bank '{}'".format(bank_id))
    bank = Bank(bank_id)
    print('a')
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                            format='[%(asctime)s] %(levelname)s: %(module)s:%(funcName)s: %(message)s',
                            filename=f"log/log_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    logging.info("8 888888888o   8 8888888888            .8.    8888888 8888888888 ,o888888o.    8 888888888o 8888888 8888888888 ")
    logging.info("8 8888    `88. 8 8888                 .888.         8 8888      8888     `88.  8 8888    `88.     8 8888       ")
    logging.info("8 8888     `88 8 8888                :88888.        8 8888   ,8 8888       `8. 8 8888     `88     8 8888       ")
    logging.info("8 8888     ,88 8 8888               . `88888.       8 8888   88 8888           8 8888     ,88     8 8888       ")
    logging.info("8 8888.   ,88' 8 888888888888      .8. `88888.      8 8888   88 8888           8 8888.   ,88'     8 8888       ")
    logging.info("8 8888888888   8 8888             .8`8. `88888.     8 8888   88 8888           8 888888888P'      8 8888       ")
    logging.info("8 8888    `88. 8 8888            .8' `8. `88888.    8 8888   88 8888   8888888 8 8888             8 8888       ")
    logging.info("8 8888      88 8 8888           .8'   `8. `88888.   8 8888   `8 8888       .8' 8 8888             8 8888       ")
    logging.info("8 8888    ,88' 8 8888          .888888888. `88888.  8 8888      8888     ,88'  8 8888             8 8888       ")
    logging.info("8 888888888P   8 888888888888 .8'       `8. `88888. 8 8888       `8888888P'    8 8888             8 8888       ")

    try:
        main()
    except Exception as e:
        logging.critical(f'Catched exception {e}')
        raise Exception