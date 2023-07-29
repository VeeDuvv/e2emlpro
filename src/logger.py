## for any exection that happens in the application, the logger will log the error in the log file

import logging
import os
import sys
from datetime import datetime
from exception import CustomException

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log" # get the log file name
logs_path = os.path.join(os.getcwd(), "logs") # get the logs folder path
os.mkdir(logs_path) if not os.path.exists(logs_path) else None # create the logs folder if it does not exist
LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE) # get the logs file path


# change loggin.INFO to logging.DEBUG to log all the debug messages
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s') # configure the logger

# if __name__ == "__main__":
#     try:
#         1/0
#     except Exception as e:
#         logging.info("Raise custom exception")
#         raise CustomException(e, sys.exc_info())

#     logging.info("This is an info message")
#     logging.debug("This is a debug message")
#     logging.warning("This is a warning message")
#     logging.error("This is an error message")
#     logging.critical("This is a critical message")