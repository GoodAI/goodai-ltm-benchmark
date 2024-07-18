# app/utils/logging.py
import logging
from logging.handlers import RotatingFileHandler
from app.config import config

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Setup loggers
main_logger = setup_logger('main', config.LOG_FILE_MAIN, level=config.LOG_LEVEL)
custom_logger = setup_logger('custom', config.LOG_FILE_CUSTOM, level=config.LOG_LEVEL)
chat_logger = setup_logger('chat', config.LOG_FILE_CHAT, level=config.LOG_LEVEL)

def get_logger(name: str):
    if name == 'chat':
        return chat_logger
    elif name == 'custom':
        return custom_logger
    else:
        return main_logger