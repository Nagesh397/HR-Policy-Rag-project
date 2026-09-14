import os
import yaml
import requests
from qdrant_client import QdrantClient
from .utils.vectorstore import rerank_points


#Ollama API endpoints (override with OLLAMA_BASE_URL, e.g. http://host.docker.internal:11434 in Docker)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
EMBED_URL = f"{OLLAMA_BASE_URL}/api/embeddings"

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# Convert user question into embedding
def embed_query(query, model):

    response = requests.post(
        EMBED_URL,
        json={
            "model": model,
            "prompt": query
        }
    )

    data = response.json()

    if "embedding" not in data:
        raise Exception(f"Ollama embedding error: {data}")

    return data["embedding"]


# Ask LLM with retrieved context
def ask_llm(context, question, model):

    prompt = f"""
You are an HR assistant.

Use only the provided context to answer the question. If the context does not contain
the answer, say that the approved policy documents do not provide that information.
Do not invent policy details. Include the source markers [1], [2], etc. for claims.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    if "response" not in data:
        raise Exception(f"Ollama generation error: {data}")

    return data["response"]


def main():

    config = load_config()

    # Connect to Qdrant (env vars override config.yaml, useful inside Docker)
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", config["qdrant"]["host"]),
        port=int(os.getenv("QDRANT_PORT", config["qdrant"]["port"]))
    )

    # User question
    question = input("\nAsk a question: ")

    # Convert question to embedding
    query_vector = embed_query(
        question,
        config["ollama"]["embedding_model"]
    )

    # Vector search in Qdrant
    results = client.query_points(
        collection_name=config["qdrant"]["collection_name"],
        query=query_vector,
        limit=5
    )
    ranked_points = rerank_points(results.points, question, limit=3)

    # Keep source metadata in the prompt so the answer can cite its evidence.
    context_parts = []
    for index, point in enumerate(ranked_points, start=1):
        payload = point.payload
        page = payload.get("page_number")
        location = f"page {page}" if page else "document section"
        context_parts.append(
            f"[{index}] {payload.get('document_title', payload.get('source_file', 'Unknown source'))}, "
            f"{location}:\n{payload['text']}"
        )
    context = "\n\n".join(context_parts)

    print("\nRetrieved Context:\n")
    print(context)

    # Ask LLM
    answer = ask_llm(
        context,
        question,
        config["ollama"]["llm_model"]
    )

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:\n")
    for index, point in enumerate(ranked_points, start=1):
        payload = point.payload
        page = payload.get("page_number")
        location = f"page {page}" if page else "document section"
        print(f"[{index}] {payload.get('document_title', payload.get('source_file', 'Unknown source'))}, {location}")


if __name__ == "__main__":
    main()