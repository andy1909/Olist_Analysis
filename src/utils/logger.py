import logging
import os
from datetime import datetime

def setup_logger(log_dir='logs'):
    """Set up application logging with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger('olist_analysis')
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(
            logging.Formatter('[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s')
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
