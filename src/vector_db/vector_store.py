import chromadb
from chromadb.config import Settings
import os
from .pdf_processor import PDFProcessor

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents"):
        self.client = chromadb.Client(Settings(persist_directory="./chroma_db"))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.processor = PDFProcessor()

    def index_pdfs(self, pdf_path: str):
        if os.path.isfile(pdf_path):
            print(f"Indexing {pdf_path}...")
            chunks, embeddings, metadatas = self.processor.process_pdf(pdf_path)
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=[f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
            )
        elif os.path.isdir(pdf_path):
            if not os.path.exists(pdf_path):
                print(f"Directory {pdf_path} not found. Skipping indexing.")
                return
            for filename in os.listdir(pdf_path):
                if filename.endswith(".pdf"):
                    pdf_file_path = os.path.join(pdf_path, filename)
                    print(f"Indexing {pdf_file_path}...")
                    chunks, embeddings, metadatas = self.processor.process_pdf(pdf_file_path)
                    self.collection.add(
                        documents=chunks,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=[f"{filename}_{i}" for i in range(len(chunks))]
                    )
        else:
            raise ValueError(f"'{pdf_path}' is neither a file nor a directory")
        print("PDF indexing complete.")

    def get_collection(self):
        return self.collection

    def get_embedding_model(self):
        return self.processor.embedding_model