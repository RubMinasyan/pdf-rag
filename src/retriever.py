"""
TF-IDF retrieval: rank chunks against a user question.
"""

import math
import re
from collections import Counter


_STOP_WORDS = frozenset(
    "a an and are as at be by can could did do does for from had has have how i "
    "in is it its me of on or our please should so that the their them there "
    "they this to was we were what when where which who why will with would you your".split()
)


def _tokenize(text: str) -> list[str]:
    """Return lowercase word and number tokens, excluding common filler words."""
    return [
        token for token in re.findall(r"[^\W_]+", text.casefold())
        if token not in _STOP_WORDS
    ]


def _compute_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    """Inverse document frequency across chunks."""
    n = len(corpus_tokens)
    df: dict[str, int] = {}
    for tokens in corpus_tokens:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1
    return {word: math.log((n + 1) / (count + 1)) + 1 for word, count in df.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """TF-IDF vector for a single document / query."""
    tf = Counter(tokens)
    length = len(tokens) or 1
    return {word: (count / length) * idf.get(word, 1.0) for word, count in tf.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[w] * b[w] for w in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class TFIDFRetriever:
    """Build a TF-IDF index over text chunks, then retrieve the top-k."""

    def __init__(self, chunks: list[dict]):
        """
        Args:
            chunks: list of {'text': str, 'pages': list[int], ...}
        """
        self.chunks = chunks
        corpus_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
        self._idf = _compute_idf(corpus_tokens)
        self._vectors = [
            _tfidf_vector(tokens, self._idf) for tokens in corpus_tokens
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return the top_k most relevant chunks for the query.

        Each result is a dict with 'text', 'pages', and 'score'.
        """
        query_tokens = _tokenize(query)
        query_vec = _tfidf_vector(query_tokens, self._idf)

        scored = []
        for idx, doc_vec in enumerate(self._vectors):
            score = _cosine_similarity(query_vec, doc_vec)
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scored[:top_k]:
            results.append(
                {
                    "text": self.chunks[idx]["text"],
                    "pages": self.chunks[idx]["pages"],
                    "score": score,
                }
            )
        return results
