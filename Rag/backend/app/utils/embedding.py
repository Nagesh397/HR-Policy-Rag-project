import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/embeddings"

def generate_embeddings(chunks, model):
    
    vectors = []
    
    for chunk in chunks:
        text = chunk["text"] if isinstance(chunk, dict) else chunk

        response = requests.post(
            OLLAMA_URL,
            json={
                "model":model,
                "prompt": text
            }
        )
        
        data = response.json()

        if "embedding" not in data:
            print("Ollama response error:", data)
            raise Exception("embedding not returned")
        
        vectors.append(data["embedding"])
        
    return vectors    