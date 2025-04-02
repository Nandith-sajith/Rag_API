# RAG_API/Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for nltk, chromadb, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (optimization for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt

# Copy the entire project
COPY . .

# Set environment variables (optional, overridden by .env)
ENV PYTHONUNBUFFERED=1

# Command to run the app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]