# app/utils/logging.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from app.config import config

class UnicodeHandler(RotatingFileHandler):
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.stream = None  # Initialize stream as None

    def emit(self, record):
        if self.stream is None:
            self.stream = self._open()  # Open the stream if it's not already open
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            msg = record.getMessage().encode('ascii', 'ignore').decode('ascii')
            formatted = self.format(record)
            formatted = formatted.replace(record.getMessage(), msg)
            stream = self.stream
            stream.write(formatted + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = UnicodeHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add unicode_safe_log method
    def unicode_safe_log(self, level, msg, *args, **kwargs):
        try:
            if level == logging.INFO:
                self.info(msg, *args, **kwargs)
            elif level == logging.WARNING:
                self.warning(msg, *args, **kwargs)
            elif level == logging.ERROR:
                self.error(msg, *args, **kwargs)
            elif level == logging.DEBUG:
                self.debug(msg, *args, **kwargs)
            else:
                self.log(level, msg, *args, **kwargs)
        except UnicodeEncodeError:
            safe_msg = msg.encode('ascii', 'ignore').decode('ascii')
            if level == logging.INFO:
                self.info(safe_msg, *args, **kwargs)
            elif level == logging.WARNING:
                self.warning(safe_msg, *args, **kwargs)
            elif level == logging.ERROR:
                self.error(safe_msg, *args, **kwargs)
            elif level == logging.DEBUG:
                self.debug(safe_msg, *args, **kwargs)
            else:
                self.log(level, safe_msg, *args, **kwargs)

    logger.unicode_safe_log = unicode_safe_log.__get__(logger)
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