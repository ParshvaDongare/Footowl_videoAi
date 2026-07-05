from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from app.utils import tokenize


@dataclass
class HashingEmbeddingFunction:
    dimensions: int = 128

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in input]

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            idx = abs(hash(token)) % self.dimensions
            vec[idx] += 1.0
        norm = sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

