from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticSearchEngine:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def cosine_similarity(self, query_embedding: NDArray[np.float32], candidate_embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        if candidate_embeddings.size == 0:
            return np.array([], dtype=np.float32)

        query_norm = np.linalg.norm(query_embedding)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)

        if query_norm == 0 or np.any(candidate_norms == 0):
            return np.zeros(candidate_embeddings.shape[0], dtype=np.float32)

        numerator = candidate_embeddings @ query_embedding
        denominator = candidate_norms * query_norm
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32), where=denominator != 0)
