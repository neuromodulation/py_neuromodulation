import logging

# include the filename in the log output
# Configure the logger
logger = logging.getLogger("PynmLogger")
logger.setLevel(logging.INFO)

# Create a file handler and set its level to DEBUG
# change mode to "a" to append to the file
file_handler = logging.FileHandler("logfile_pynm.log", mode="w")
file_handler.setLevel(logging.INFO)

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(name)s:%(filename)s:%(message)s"
)
file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
# logger.addHandler(console_handler)
