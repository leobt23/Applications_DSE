import logging

# TODO Clean up log.txt file when starting the app

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)-15s %(message)s")

# Add a FileHandler to save log messages to a file
file_handler = logging.FileHandler("log.txt")
file_handler.setFormatter(formatter)
app_logger.addHandler(file_handler)

# Add a StreamHandler to print log messages to the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
app_logger.addHandler(stream_handler)
