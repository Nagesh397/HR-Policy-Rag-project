# Employee Policy Assistant

A simple full-stack RAG application that answers HR policy questions from approved, backend-owned documents and returns source citations.

## How It Works

```text
Fixed policy PDFs -> extraction -> cleaning -> chunking -> Ollama embeddings
-> Qdrant search -> reranking -> Ollama answer -> frontend response with sources
```

There is no document upload feature. The backend indexes the documents in
`data/knowledge_base` automatically when the Qdrant collection is empty.

## Requirements

- Docker Desktop
- Ollama running at `http://localhost:11434`
- Ollama models: `nomic-embed-text` and `llama3.2:3b`

## Run the Application

From the `Rag` directory:

```powershell
docker compose up --build
```

Open the frontend at `http://localhost:3000`.

The backend API is available at `http://localhost:8000` and Swagger documentation is available at `http://localhost:8000/docs`.

Stop the services with:

```powershell
docker compose down
```

## API Endpoints

```text
GET  /health
POST /api/chat
GET  /api/chats
GET  /api/chats/{chat_id}
```

Create a chat message:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/chat `
	-Method Post `
	-ContentType "application/json" `
	-Body '{"question":"What is the work from home policy?"}'
```

Chat history is currently stored in backend memory and resets when the backend restarts.

## Project Structure

```text
Rag/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI routes and chat history
│   │   ├── ingestion.py         # Indexes approved documents
│   │   ├── query.py             # Embeddings, retrieval, and LLM calls
│   │   └── utils/               # Loading, chunking, embeddings, Qdrant helpers
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── knowledge_base/          # Fixed approved policy PDFs
│   └── evaluation/              # Questions and evaluation script
├── frontend/
│   ├── index.html
│   └── Dockerfile
├── config.yaml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Evaluation

With Ollama and Qdrant running, execute from the `Rag` directory:

```powershell
python data/evaluation/test_evaluate.py
```

The evaluation script reads `data/evaluation/test_questions.yaml` and writes its generated report to `data/evaluation/evaluation_results.json`.
