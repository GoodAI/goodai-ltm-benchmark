from functools import wraps
import time
from src.utils.structured_logging import get_logger
import asyncio

performance_logger = get_logger("performance")

def log_execution_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        performance_logger.info(f"{func.__name__} executed",
                                execution_time=execution_time,
                                function=func.__name__)
        return result
    return wrapper