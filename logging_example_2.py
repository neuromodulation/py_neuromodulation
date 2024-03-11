import logging

# include the filename in the log output
# Configure the logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.INFO)

# Create a file handler and set its level to DEBUG
file_handler = logging.FileHandler("logfile.log")
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(name)s:%(filename)s:%(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class FirstLogger:
    def __init__(self) -> None:
        pass

    def call_logging_warnining(
        self,
    ):
        logger.warning("Logger 1: Warning")

    def call_logging_error(
        self,
    ):
        try:
            raise ValueError("Logger 2: Error")
        except ValueError as e:
            # logging.exception("message")
            logger.exception(e, exc_info=True)
