import yaml
from qdrant_client import QdrantClient

from .utils.loader import load_document
from .utils.preprocessing import preprocess
from .utils.chunking import chunk_text
from .utils.embedding import generate_embeddings
from .utils.vectorstore import setup_collection, ingest_chunks

import os

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base")


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()

    knowledge_base = os.path.abspath(KNOWLEDGE_BASE_PATH)
    paths = [
        os.path.join(knowledge_base, name)
        for name in os.listdir(knowledge_base)
        if name.lower().endswith((".pdf", ".md", ".txt"))
    ]
    if not paths:
        raise FileNotFoundError(f"No supported documents found in {knowledge_base}")

    chunks = []
    for path in sorted(paths):
        print(f"Loading {os.path.basename(path)}...")
        for page in load_document(path):
            cleaned = preprocess(page["text"])
            chunks.extend(chunk_text(cleaned, {
                "source_file": os.path.basename(path),
                "document_title": os.path.splitext(os.path.basename(path))[0].replace("_", " ").title(),
                "page_number": page["page_number"],
                "approved": True,
            }))

    print("Total chunks:", len(chunks))

    print("Generating embeddings via Ollama...")
    embeddings = generate_embeddings(
        chunks,
        config["ollama"]["embedding_model"]
    )

    print("Connecting Qdrant...")
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", config["qdrant"]["host"]),
        port=int(os.getenv("QDRANT_PORT", config["qdrant"]["port"]))
    )

    setup_collection(
        client,
        config["qdrant"]["collection_name"],
        config["qdrant"]["vector_size"]
    )

    print("Ingesting vectors...")
    ingest_chunks(
        client,
        config["qdrant"]["collection_name"],
        chunks,
        embeddings
    )

    print("✅ Ingestion completed")


if __name__ == "__main__":
    main()