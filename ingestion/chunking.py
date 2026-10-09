"""Semantic chunking for normalized documents from any ingestion source.

File parsing belongs in source loaders. This module only consumes text and
metadata, so PDF, ODT/ODF, and frontend-uploaded documents follow the same
chunking path after they have been converted to text.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class EmbeddingFunction(Protocol):
	"""Batch embedding callable used by the semantic boundary detector."""

	def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
		...


@dataclass(frozen=True, slots=True)
class ContentBlock:
	"""Text representation of a non-text document element."""

	content_type: str
	text: str
	metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Document:
	"""Normalized input shared by PDF, ODT/ODF, and API upload loaders."""

	text: str
	source_id: str
	metadata: Mapping[str, Any] = field(default_factory=dict)
	content_blocks: Sequence[ContentBlock | Mapping[str, Any]] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SemanticChunk:
	"""A persisted retrieval unit with stable identity and source traceability."""

	chunk_id: str
	text: str
	source_id: str
	chunk_index: int
	metadata: Mapping[str, Any] = field(default_factory=dict)
	embedding: tuple[float, ...] | None = None

	def as_record(self) -> dict[str, Any]:
		record = {
			"chunk_id": self.chunk_id,
			"text": self.text,
			"source_id": self.source_id,
			"chunk_index": self.chunk_index,
			"metadata": dict(self.metadata),
		}
		if self.embedding is not None:
			record["embedding"] = list(self.embedding)
		return record


@dataclass(frozen=True, slots=True)
class SemanticChunkerConfig:
	"""Bounds that keep chunks useful for retrieval and LLM context windows."""

	target_characters: int = 1800
	max_characters: int = 2600
	min_characters: int = 300
	similarity_threshold: float = 0.72
	overlap_sentences: int = 1
	embedding_batch_size: int = 64

	def __post_init__(self) -> None:
		if not 0 < self.min_characters <= self.target_characters <= self.max_characters:
			raise ValueError("character bounds must satisfy 0 < min <= target <= max")
		if not 0 <= self.similarity_threshold <= 1:
			raise ValueError("similarity_threshold must be between 0 and 1")
		if self.overlap_sentences < 0 or self.embedding_batch_size < 1:
			raise ValueError("overlap_sentences and embedding_batch_size must be non-negative")


_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def split_sentences(text: str) -> list[str]:
	"""Split normalized text while retaining useful short lines and headings."""

	normalized = _WHITESPACE_PATTERN.sub(" ", text).strip()
	if not normalized:
		return []
	return [part.strip() for part in _SENTENCE_PATTERN.split(normalized) if part.strip()]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
	if len(left) != len(right) or not left or not right:
		return 0.0
	left_norm = math.sqrt(sum(value * value for value in left))
	right_norm = math.sqrt(sum(value * value for value in right))
	if not left_norm or not right_norm:
		return 0.0
	return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _lexical_similarity(left: str, right: str) -> float:
	"""Deterministic fallback for local development and unavailable embeddings."""

	left_terms = set(re.findall(r"[a-z0-9]{2,}", left.lower()))
	right_terms = set(re.findall(r"[a-z0-9]{2,}", right.lower()))
	union = left_terms | right_terms
	return len(left_terms & right_terms) / len(union) if union else 0.0


def _batched(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
	for start in range(0, len(values), size):
		yield values[start : start + size]


class SemanticChunker:
	"""Create semantic chunks without assuming a particular document format.

	``embedding_function`` should batch texts and return one vector per text.
	It can be backed by a hosted model, a local model, or a test double. If it
	is omitted, lexical similarity keeps the component usable offline.
	"""

	def __init__(
		self,
		embedding_function: EmbeddingFunction | None = None,
		config: SemanticChunkerConfig | None = None,
	) -> None:
		self.embedding_function = embedding_function
		self.config = config or SemanticChunkerConfig()

	def chunk_documents(self, documents: Iterable[Document]) -> Iterator[SemanticChunk]:
		"""Stream chunks so large uploads do not need to fit in memory at once."""

		for document in documents:
			yield from self.chunk_document(document)

	def chunk_document(self, document: Document) -> Iterator[SemanticChunk]:
		content_text, content_types = self._content_text(document)
		sentences = split_sentences(content_text)
		if not sentences:
			return

		similarities = self._adjacent_similarities(sentences)
		boundaries = self._choose_boundaries(sentences, similarities)
		start = 0
		chunk_index = 0
		chunks: list[SemanticChunk] = []
		for end in boundaries:
			chunk_sentences = sentences[start:end]
			chunk_text = " ".join(chunk_sentences).strip()
			if chunk_text:
				metadata = dict(document.metadata)
				metadata.update({"chunking": "semantic", "sentence_start": start, "sentence_end": end})
				if content_types:
					metadata["content_types"] = content_types
				chunk_id = self._chunk_id(document.source_id, chunk_index, chunk_text)
				chunks.append(SemanticChunk(chunk_id, chunk_text, document.source_id, chunk_index, metadata))
				chunk_index += 1
			start = max(start + 1, end - self.config.overlap_sentences)

		if self.embedding_function is not None and chunks:
			embeddings = self._embed_texts([chunk.text for chunk in chunks])
			for chunk, embedding in zip(chunks, embeddings):
				yield SemanticChunk(
					chunk.chunk_id,
					chunk.text,
					chunk.source_id,
					chunk.chunk_index,
					chunk.metadata,
					tuple(float(value) for value in embedding),
				)
		else:
			yield from chunks

	@staticmethod
	def _content_text(document: Document) -> tuple[str, list[str]]:
		"""Combine text and searchable representations of tables and images."""

		parts = [document.text.strip()]
		content_types: list[str] = []
		for raw_block in document.content_blocks:
			if isinstance(raw_block, ContentBlock):
				block = raw_block
			else:
				block = ContentBlock(
					content_type=str(raw_block.get("content_type", "text")),
					text=str(raw_block.get("text", raw_block.get("content", ""))),
					metadata={key: value for key, value in raw_block.items() if key not in {"content_type", "text", "content"}},
				)
			block_text = block.text.strip()
			if not block_text:
				continue
			content_type = block.content_type.strip().lower() or "text"
			parts.append(f"[{content_type}] {block_text}")
			if content_type not in content_types:
				content_types.append(content_type)
		return "\n\n".join(part for part in parts if part), content_types

	def _adjacent_similarities(self, sentences: Sequence[str]) -> list[float]:
		if len(sentences) < 2:
			return []
		if self.embedding_function is None:
			return [_lexical_similarity(a, b) for a, b in zip(sentences, sentences[1:])]

		vectors = self._embed_texts(sentences)
		return [_cosine_similarity(a, b) for a, b in zip(vectors, vectors[1:])]

	def _embed_texts(self, texts: Sequence[str]) -> list[Sequence[float]]:
		if self.embedding_function is None:
			return []
		vectors: list[Sequence[float]] = []
		for batch in _batched(texts, self.config.embedding_batch_size):
			batch_vectors = list(self.embedding_function(batch))
			if len(batch_vectors) != len(batch):
				raise ValueError("embedding function must return one vector per input text")
			vectors.extend(batch_vectors)
		return vectors

	def _choose_boundaries(self, sentences: Sequence[str], similarities: Sequence[float]) -> list[int]:
		boundaries: list[int] = []
		start = 0
		for index in range(1, len(sentences) + 1):
			current_length = len(" ".join(sentences[start:index]))
			reached_limit = current_length >= self.config.target_characters
			too_large = current_length > self.config.max_characters
			semantic_break = index < len(sentences) and similarities[index - 1] < self.config.similarity_threshold
			if index == len(sentences) or too_large or (reached_limit and semantic_break):
				if current_length >= self.config.min_characters or index == len(sentences):
					boundaries.append(index)
					start = index
		if not boundaries or boundaries[-1] < len(sentences):
			boundaries.append(len(sentences))
		return boundaries

	@staticmethod
	def _chunk_id(source_id: str, chunk_index: int, text: str) -> str:
		digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
		return f"{source_id}:{chunk_index}:{digest}"


def write_jsonl(chunks: Iterable[SemanticChunk], output_path: str | Path) -> int:
	"""Write chunks incrementally and return the number of records written."""

	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	count = 0
	with path.open("w", encoding="utf-8") as output:
		for chunk in chunks:
			output.write(json.dumps(chunk.as_record(), ensure_ascii=True) + "\n")
			count += 1
	return count
