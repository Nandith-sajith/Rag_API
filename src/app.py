import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.routes.rag_query import router as rag_query_router 
from src.core.dependencies import get_executor, get_vector_store

# Initialize the FastAPI app
app = FastAPI()

ALLOWED_ORIGINS = ["*"]

# Update CORS middleware during production
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    pdf_path = "assets/Monopoly rules.pdf"
    vector_store = get_vector_store()
    vector_store.index_pdfs(pdf_path)

@app.on_event("shutdown")
async def shutdown_event():
    executor = get_executor()
    executor.shutdown(wait=True)

app.include_router(rag_query_router, prefix="/rag_query", dependencies=[Depends(get_vector_store)])