from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a small, fast embedding model.
        Good options: 
        - 'all-MiniLM-L6-v2' (fast, 384-dim embeddings)
        - 'paraphrase-MiniLM-L3-v2' (even smaller)
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """
        Embed a list of texts or a single text string.
        Returns numpy array(s).
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

    def embed_text(self, text):
        return list(self.embed([text])[0])

    def similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two embeddings.
        """
        return np.dot(emb1, emb2)

# --------------------
if __name__ == "__main__":
    embedder = LocalEmbedder()

    text1 = "Artificial Intelligence is fascinating."
    text2 = "Machine Learning is a branch of AI."

    # Generate Embeddings
    emb1 = embedder.embed(text1)[0]
    emb2 = embedder.embed(text2)[0]

    # Compute Similarity
    sim = embedder.similarity(emb1, emb2)
    print(f"Cosine Similarity: {sim:.4f}")