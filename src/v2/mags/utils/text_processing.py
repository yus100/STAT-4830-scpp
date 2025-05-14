"""
Utility functions for text processing.
"""
import re

# Common articles to ignore
ARTICLES = {"a", "an", "the"}

def normalize_text(text: str) -> str:
    """
    Normalizes text by converting to lowercase, removing articles,
    and stripping extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove punctuation (optional, depending on desired strictness)
    # text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    # Remove articles
    words = [word for word in words if word not in ARTICLES]
    # Join and strip extra spaces
    return " ".join(words).strip()

def are_texts_equivalent(text1: str, text2: str) -> bool:
    """
    Checks if two texts are equivalent after normalization.
    Ignores case, articles, and extra spaces.
    """
    return normalize_text(text1) == normalize_text(text2)