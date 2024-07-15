from functools import wraps
import time
import structlog

def log_execution_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = structlog.get_logger(func.__module__)
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} executed",
                    execution_time=execution_time,
                    function=func.__name__)
        return result
    return wrapper