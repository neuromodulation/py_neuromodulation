import logging
import logging_example_2


# from logging_example_2 import logger
logger = logging.getLogger("MyLogger")


class SecondLogger:
    def __init__(self) -> None:
        pass

    def call_logging_warnining(
        self,
    ):
        logger.warning("Logger 2: Warning")

    def call_logging_error(
        self,
    ):
        try:
            raise ValueError("Logger 2: Error")
        except ValueError as e:
            # logging.exception("message")
            logger.exception(e, exc_info=True)


if __name__ == "__main__":

    log_1 = logging_example_2.FirstLogger()
    log_1.call_logging_warnining()

    # log_1 has a logger that writes to a file, how can the logger in this file be used to write to the same file?

    log_2 = SecondLogger()
    log_2.call_logging_warnining()
    log_2.call_logging_error()

    logger.info("This is a test")

    # write now the logging output to a file
