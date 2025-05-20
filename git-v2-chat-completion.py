# chat_completion.py
import os
import pickle
import faiss
import numpy as np
import openai
import time
from openai import RateLimitError, OpenAIError

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")


# Load the saved FAISS index and metadata
faiss_index = faiss.read_index("index.faiss")
with open("index_meta.pkl", "rb") as f:
    index_meta = pickle.load(f)

# Reuse the same embedding function from build_index.py
def get_embedding(text: str, model="text-embedding-ada-002", max_retries=5) -> np.ndarray:
    for i in range(max_retries):
        try:
            resp = openai.embeddings.create(
                model=model,
                input=[text]
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)

        except RateLimitError:
            wait = 2 ** i
            print(f"🚦 RateLimitError, retrying in {wait}s…")
            time.sleep(wait)

        except OpenAIError as e:
            print("❌ OpenAI error:", e)
            raise
    raise RuntimeError(f"Failed to get embedding after {max_retries} retries")

# Simple chat loop
def chat():
    print("🤖 Ask me anything about the Canara Bank website! (type 'exit' to quit)")
    while True:
        query = input("\nYou: ")
        if query.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break

        q_emb = get_embedding(query).reshape(1, -1)
        D, I = faiss_index.search(q_emb, k=1)  # You can increase k for top-N
        best_match = index_meta[I[0][0]]

        print(f"\n🔗 Best match: {best_match['title']}")
        print(f"🌐 URL: {best_match['url']}")
        print(f"📐 Cosine similarity score: {1 - D[0][0]:.4f}")

if __name__ == "__main__":
    chat()
