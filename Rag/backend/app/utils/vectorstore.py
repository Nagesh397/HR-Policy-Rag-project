from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

def setup_collection(client, collection_name, vector_size):

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )


def ingest_chunks(client, collection_name, chunks, embeddings):

    points = []

    for chunk, vector in zip(chunks, embeddings):
        text = chunk["text"] if isinstance(chunk, dict) else chunk
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, **metadata}
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )


def rerank_points(points, question, limit=3):
    """Use simple term overlap to improve ordering after vector retrieval."""
    question_terms = set(question.lower().split())

    def score(point):
        text_terms = set(point.payload.get("text", "").lower().split())
        return len(question_terms & text_terms)

    return sorted(points, key=score, reverse=True)[:limit]