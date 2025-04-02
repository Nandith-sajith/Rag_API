# src/core/utils.py
import time
import logging
from typing import Callable, Any, List
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def measure_time(func: Callable) -> Callable:
    """Decorator to measure execution time of an async function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        return result
    return wrapper

def calculate_confidence_score(context: str, answer: str, keywords: List[str]) -> float:
    """Simple confidence score based on keyword overlap and context presence."""
    if not context or "No relevant context" in context:
        return 0.0
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())
    keyword_matches = sum(1 for kw in keywords if kw.lower() in answer.lower())
    overlap = len(context_words.intersection(answer_words)) / len(context_words) if context_words else 0
    return min(1.0, overlap + (keyword_matches * 0.1))  # Cap at 1.0