# src/utils/api_utils.py

import time
import asyncio
from functools import wraps
from cachetools import TTLCache

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    async def wait(self):
        now = time.time()
        self.calls = [call for call in self.calls if call > now - self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] - (now - self.period)
            await asyncio.sleep(sleep_time)
        self.calls.append(time.time())

def rate_limited(max_calls, period):
    limiter = RateLimiter(max_calls, period)
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await limiter.wait()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def exponential_backoff(max_retries=5, base_delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:  # Replace with specific exception if needed
                    if retries == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** retries)
                    print(f"API call failed. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    retries += 1
        return wrapper
    return decorator

# Simple cache with TTL
cache = TTLCache(maxsize=100, ttl=300)  # Cache up to 100 items for 5 minutes

def cached(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key in cache:
            return cache[key]
        result = await func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper