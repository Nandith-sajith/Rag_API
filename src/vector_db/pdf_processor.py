from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Tuple
import os

nltk.data.path.append(os.path.join(os.path.dirname(__file__), '../../nltk_data'))

class PDFProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract text with page numbers as metadata."""
        reader = PdfReader(pdf_path)
        pages = []
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append({"text": text, "page": str(page_num)})
        return pages

    def chunk_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Chunk text into smaller pieces."""
        sentences = sent_tokenize(text) #nltk
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def process_pdf(self, pdf_path: str) -> Tuple[List[str], List[float], List[Dict[str, str]]]:
        """Process PDF and return chunks, embeddings, and metadata."""
        pages = self.extract_text_from_pdf(pdf_path)
        chunks = []
        metadatas = []
        
        for page in pages:
            page_chunks = self.chunk_text(page["text"])
            for chunk in page_chunks:
                chunks.append(chunk)
                metadatas.append({"page": page["page"]})  # Metadata with page number
        
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False).tolist()
        return chunks, embeddings, metadatas