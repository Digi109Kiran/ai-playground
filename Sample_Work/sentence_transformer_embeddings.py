from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading SentenceTransformer model: {model_name}...")
        print(f"Model name passed: {model_name}")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded successfully.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of documents (strings) into vectors."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string into a vector."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
