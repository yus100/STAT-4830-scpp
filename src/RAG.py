from Embedding import Embedding
from LGK import LGK
from Llama import LLM
from config.config import Config
from Graph import Node, Edge, Graph
from Memory import Memory, MemoryBlock

import numpy as np

class Similarity():
    @staticmethod
    def cosine_similarity(a, b):
        """
        Computes the cosine similarity between two vectors.
        """
        # Convert lists to numpy arrays if needed
        a = np.array(a) if isinstance(a, list) else a
        b = np.array(b) if isinstance(b, list) else b
        
        # Reshape vectors to ensure proper alignment
        a = a.reshape(-1)
        b = b.reshape(-1)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray):
        """
        Computes the Euclidean distance between two vectors.
        """
        return np.linalg.norm(a - b)
    

class RAG():

    _config = Config()

    @staticmethod
    async def multiquery(query: str, llm: LLM) -> list[str]:
        """
        Queries the LLM with a given query and returns the response.
        """
        
        full_prompt = RAG()._config.get_prompt("multiquery")
        # replace "{question}" with the query DO NOT USE .format() method
        full_prompt = full_prompt.replace("{question}", query)

        return (await llm.query_json(full_prompt))["rephrased"]

    @staticmethod
    async def find_documents(queries: list[str], db: LGK, top_k: int = 5, allow_duplicates: bool = False, similarity_metric = Similarity.cosine_similarity):
        """
        Finds the top-k most similar documents to each query.
        """
        # Encode the queries and documents
        encoded_queries = []
        for query in queries:
            encoded_queries.append(await Embedding().embed_document(query))
        
        nodes: list[Memory] = db.get_all_nodes()

        # Find the top-k most similar documents for each query
        query_results = []
        for query_embedding in encoded_queries:
            # Calculate similarity scores for each node
            node_scores = {}
            for node in nodes:
                # Get all embeddings for this node
                node_embeddings = await node._memoryblock.get_embeddings()
                # Calculate similarity with each embedding and sum them
                total_similarity = sum(similarity_metric(query_embedding, emb) for emb in node_embeddings)
                node_scores[node] = total_similarity
            
            # Sort nodes by score and get top_k
            sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
            query_results.append([node for node, score in sorted_nodes[:top_k]])
        
        # Aggregate results across queries
        if not allow_duplicates:
            # Count frequency of each node across all query results
            node_frequency = {}
            for query_top_k in query_results:
                for node in query_top_k:
                    node_frequency[node] = node_frequency.get(node, 0) + 1
            
            # Sort by frequency and take top_k
            final_nodes = sorted(node_frequency.items(), key=lambda x: x[1], reverse=True)
            return [node for node, freq in final_nodes[:top_k]]
        else:
            # If duplicates allowed, just take top_k from first query's results
            return query_results[0][:top_k]