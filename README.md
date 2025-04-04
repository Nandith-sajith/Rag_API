# RAG API with FastAPI and Groq

This project is a FastAPI application that implements a Retrieval-Augmented Generation (RAG) system, leveraging Groq's AI capabilities to query documents using natural language. This README explains how to run the application locally using Docker Compose.

## Prerequisites

- **Docker** and **Docker Compose** installed and running.
- A `.env` file containing your Groq API key.

## Local Setup Workflow

Follow these steps to get your application running locally:

### 1. Clone the Repository

Clone this repository and navigate into the project directory:

```bash
git clone https://github.com/Nandith-sajith/Rag_API.git
cd Rag_API
```

### 2. Set Up the `.env` File

Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=<your-groq-api-token>
```

### 3. Start the Application

Run the following command to start the application using Docker Compose:

```bash
docker-compose up --build -d
```

### 4. Test the API

Once the application is running, you can test it by sending a request using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/rag_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "If Player A mortgages a property, then lands on an unowned property but does not buy it, and Player B later lands on the same property, can Player B auction it and use a Get Out of Jail Free card as part of the bid?"
  }'
```
