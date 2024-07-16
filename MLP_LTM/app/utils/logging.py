import logging
from logging.handlers import RotatingFileHandler
from app.config import config
import os

def setup_logging():
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(config.LOG_LEVEL)  # Set to DEBUG to capture all levels

    # Create handlers
    file_handler = RotatingFileHandler(
        config.LOG_FILE, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_logger(name: str):
    return logging.getLogger(name)