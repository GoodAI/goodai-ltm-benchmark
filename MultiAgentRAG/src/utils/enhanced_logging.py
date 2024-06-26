import time
import logging
from functools import wraps

def log_execution_time(logger):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        return wrapper
    return decorator

class DatabaseLogger:
    def __init__(self, logger):
        self.logger = logger
        file_handler = logging.FileHandler('database.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def log_query(self, query, params=None):
        self.logger.debug(f"Executing query: {query}, Params: {params}")

    def log_memory_size(self, memory_size):
        self.logger.info(f"Memory size: {memory_size} bytes")

    def log_access(self, memory_id):
        self.logger.info(f"Accessed memory: {memory_id}")