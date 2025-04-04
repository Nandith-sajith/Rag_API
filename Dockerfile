FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download punkt_tab to the correct location
RUN mkdir -p /nltk_data && \
    python -c "import nltk; nltk.download('punkt_tab', download_dir='/nltk_data')"

ENV NLTK_DATA=/nltk_data

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]