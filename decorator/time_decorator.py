import time
from utils.logger_utils import logger
from functools import wraps

def timeit(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} executed in {(end - start):.4f} seconds")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} executed in {(end - start):.4f} seconds")
        return result
    
    return async_wrapper if hasattr(func, "__call__") and callable(func) and func.__code__.co_flags & 0x80 else sync_wrapper
        