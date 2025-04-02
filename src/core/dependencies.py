from concurrent.futures import ThreadPoolExecutor
from src.vector_db.vector_store import VectorStore
from cachetools import LRUCache
from src.core.models import PromptResponse  

# Singleton executor instance
executor = ThreadPoolExecutor(max_workers=4)

# Singleton vector store instance
vector_store = VectorStore()

# Singleton cache instance (max 100 entries)
cache = LRUCache(maxsize=100)

def get_executor() -> ThreadPoolExecutor:
    return executor

def get_vector_store() -> VectorStore:
    return vector_store

def get_cache() -> LRUCache:
    return cache