"""OpenAI embedding adapter used by the ingestion pipeline."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class OpenAIEmbeddingFunction:
    """Batch text embedding function compatible with SemanticChunker."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model = model
        self.batch_size = batch_size

        if client is not None:
            self.client = client
            return

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from .env")

        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "Install the openai package first."
            ) from error

        self.client = OpenAI(api_key=resolved_api_key)

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            items = sorted(response.data, key=lambda item: item.index)

            if len(items) != len(batch):
                raise ValueError(
                    "Embedding provider must return one vector per input text"
                )

            vectors.extend(list(item.embedding) for item in items)

        return vectors