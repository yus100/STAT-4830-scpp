"""
Utility functions for mathematical operations.
"""
from typing import List
import math

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Computes the cosine similarity between two vectors.
    Assumes vectors are of the same length and non-zero.
    """
    if len(vec1) != len(vec2) or not vec1 or not vec2:
        # Or raise ValueError for more specific error handling
        return 0.0

    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v**2 for v in vec1))
    magnitude2 = math.sqrt(sum(v**2 for v in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0  # Or handle as an error/special case

    return dot_product / (magnitude1 * magnitude2)