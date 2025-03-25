import configparser
import logging
import sys

# Configure the project logger.
# This configuration is done only once at the start of the program.
logger = logging.getLogger("EW2L_logger")

# Retrieve variables from config.ini.
config = configparser.ConfigParser()
config.read("config.ini")

if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if config["LOGGER"]["level"] == "INFO":
        logger.setLevel(logging.INFO)
    elif config["LOGGER"]["level"] == "DEBUG":
        logger.setLevel(logging.DEBUG)


def get_logger() -> logging.Logger:
    """
    Returns the configured project logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    return logger
