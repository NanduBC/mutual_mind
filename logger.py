import logging
import sys

LOG_LEVEL=logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout)

def get_logger(log_name):
    '''
    Returns a logger with `log_name` as name

    Parameters:
    ----------
    log_name: Name of the logger
    '''
    logger = logging.getLogger(log_name)

    if len(logger.handlers) < 1:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        consolehandler = logging.StreamHandler(sys.stdout)
        consolehandler.setLevel(LOG_LEVEL)
        consolehandler.setFormatter(formatter)

        logger.propagate = False
        logger.addHandler(consolehandler)
    return logger