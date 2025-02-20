import asyncio
from typing import List, Callable, Awaitable
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy

# Load the spacy model. This assumes you have a model like 'en_core_web_sm' installed.
# If not, you'll need to run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class Chunker:
    """
    Base class for chunking text.
    """
    def chunk_text(self, text: str) -> List[str]:
        raise NotImplementedError("Subclasses must implement chunk_text method")

class SentenceChunker(Chunker):
    """
    Chunks text into sentences using spaCy.
    """
    def chunk_text(self, text: str) -> List[str]:
        doc = nlp(text)
        return [sent.text for sent in doc.sents]

class FixedSizeChunker(Chunker):
    """
    Chunks text into fixed-size chunks.
    """
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

class AsIsChunker(Chunker):
    """
    Chunks text as-is.
    """
    def chunk_text(self, text: str) -> List[str]:
        return [text]

class Embedding:
    """
    Async document embedding system using Sentence Transformers.
    """
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 chunker: Chunker = AsIsChunker()):
        """
        Initializes async embedding system.

        Args:
            model_name (str): Sentence Transformers model name
            chunker (Chunker): Text chunking strategy
        """
        self.model = SentenceTransformer(model_name)
        self.chunker = chunker

    async def embed_chunk(self, chunk: str) -> np.ndarray:
        """
        Async embed a single text chunk.
        """
        return await asyncio.to_thread(self.model.encode, chunk)

    async def embed_document(self, document: str) -> List[np.ndarray]:
        """
        Async process document through chunking and embedding.
        """
        chunks = self.chunker.chunk_text(document)
        return await asyncio.gather(*[self.embed_chunk(c) for c in chunks])

    async def embed_documents(self, documents: List[str]) -> List[List[np.ndarray]]:
        """
        Async process multiple documents with parallel execution.
        """
        return await asyncio.gather(*[self.embed_document(d) for d in documents])

if __name__ == '__main__':
    async def main():
        # Example usage
        embedder = Embedding()
        
        docs = [
            "The quick brown fox jumps over the lazy dog.",
            "Sentence transformers provide meaningful sentence embeddings.",
            "Async programming enables efficient parallel execution."
        ]

        # Embed all documents concurrently
        embeddings = await embedder.embed_documents(docs)
        
        # Display results
        for i, doc_embeddings in enumerate(embeddings):
            print(f"Document {i+1} has {len(doc_embeddings)} chunks")
            print(f"First embedding shape: {doc_embeddings[0].shape}\n")

    asyncio.run(main())
