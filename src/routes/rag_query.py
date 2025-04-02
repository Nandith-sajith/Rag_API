# src/routes/rag_query.py
from fastapi import APIRouter, status, Depends
from src.core.models import PromptRequest, PromptResponse  # Import from models.py
from src.vector_db.vector_store import VectorStore
from src.core.prompt_engine import PromptEngine
from src.core.utils import measure_time
from src.core.dependencies import get_executor, get_cache
import re
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from cachetools import LRUCache

router = APIRouter()
prompt_engine = PromptEngine()

def extract_keywords(query: str) -> List[str]:
    stop_words = {"the", "is", "are", "of", "to", "and", "in", "on", "with"}
    words = re.findall(r'\w+', query.lower())
    return [word for word in words if len(word) > 2 and word not in stop_words]

async def async_encode(embedding_model, query: str, executor: ThreadPoolExecutor) -> List[float]:
    return await asyncio.to_thread(embedding_model.encode, [query], convert_to_tensor=False)

async def async_query(collection, query_embedding: List[float], executor: ThreadPoolExecutor) -> dict:
    return await asyncio.to_thread(
        collection.query,
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

@router.post("", response_model=PromptResponse, status_code=status.HTTP_200_OK)
@measure_time
async def process_prompt(
    request: PromptRequest,
    vector_store: VectorStore = Depends(),
    executor: ThreadPoolExecutor = Depends(get_executor),
    cache: LRUCache = Depends(get_cache)
):
    # Check cache first
    query = request.query.strip().lower()  # Normalize query for consistency
    if query in cache:
        return cache[query]

    # Process query if not cached
    collection = vector_store.get_collection()
    embedding_model = vector_store.get_embedding_model()

    query_embedding = (await async_encode(embedding_model, request.query, executor)).tolist()[0]
    keywords = extract_keywords(request.query)
    vector_results = await async_query(collection, query_embedding, executor)

    documents = vector_results.get("documents", [[]])[0]
    metadatas = vector_results.get("metadatas", [[]])[0] or [{}] * len(documents)
    distances = vector_results.get("distances", [[]])[0]

    hybrid_results = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        keyword_score = sum(1 for kw in keywords if kw.lower() in doc.lower())
        hybrid_score = (1 - dist) + (keyword_score * 0.1)
        hybrid_results.append({
            "document": doc,
            "metadata": meta,
            "score": hybrid_score
        })

    hybrid_results = sorted(hybrid_results, key=lambda x: x["score"], reverse=True)[:5]

    retrieved_chunks = []
    for result in hybrid_results:
        doc = result["document"]
        meta = result["metadata"]
        meta_str = f"(Page {meta['page']})" if "page" in meta else ""
        chunk = f"{doc} {meta_str}".strip()
        retrieved_chunks.append(chunk)

    context = " ".join(retrieved_chunks).strip() if retrieved_chunks else "No relevant context found."

    answer, confidence = await prompt_engine.generate_answer(request.query, context, keywords)
    evaluation = prompt_engine.evaluate_response(answer, context, keywords)

    # Create response and cache it
    response = PromptResponse(answer=answer, confidence=confidence, evaluation=evaluation)
    cache[query] = response
    return response