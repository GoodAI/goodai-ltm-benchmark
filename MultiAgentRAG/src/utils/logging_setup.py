import logging
import os
import socket
from datetime import datetime

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    container_id = socket.gethostname() if is_running_in_docker() else 'local'
    log_directory = f'logs/{timestamp}_{container_id}'

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    master_logger = logging.getLogger('master')
    master_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(log_directory, "master.log"))
    file_handler.setFormatter(log_formatter)
    master_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    master_logger.addHandler(console_handler)

    chat_logger = logging.getLogger('chat')
    chat_logger.setLevel(logging.DEBUG)
    chat_file_handler = logging.FileHandler(os.path.join(log_directory, 'chat.log'))
    chat_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    chat_logger.addHandler(chat_file_handler)

    memory_logger = logging.getLogger('memory')
    memory_logger.setLevel(logging.DEBUG)
    memory_file_handler = logging.FileHandler(os.path.join(log_directory, 'memory.log'))
    memory_file_handler.setFormatter(log_formatter)
    memory_logger.addHandler(memory_file_handler)

    database_logger = logging.getLogger('database')
    database_logger.setLevel(logging.DEBUG)
    database_file_handler = logging.FileHandler(os.path.join(log_directory, 'database.log'))
    database_file_handler.setFormatter(log_formatter)
    database_logger.addHandler(database_file_handler)

    master_logger.info(f"Logging setup complete. Log directory: {log_directory}")
    return master_logger, chat_logger, memory_logger, database_logger

def is_running_in_docker() -> bool:
    return os.path.exists('/.dockerenv')