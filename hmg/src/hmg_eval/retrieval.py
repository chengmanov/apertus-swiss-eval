"""BM25 retrieval over Rz chunks.

Deliberately simple and fully transparent: lowercase alphanumeric tokens, no
embeddings, no external services. The retrieval method is part of the
published methodology; anyone can reproduce the exact ranking.
"""

import re

from rank_bm25 import BM25Okapi

from .corpus import Chunk

TOKEN = re.compile(r"[a-z0-9äöüéèàâçß]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(f"{c.section} {c.text}") for c in chunks])

    def top(self, query: str, k: int = 5) -> list[Chunk]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.chunks[i] for i in ranked[:k]]
