from __future__ import annotations

import dataclasses
from dataclasses import field
from enum import Enum
from typing import Callable, TypedDict


class StrEnum(str, Enum):
    """Enum where members are also (and must be) strings"""

    def __str__(self) -> str:
        return str(self.value)


@dataclasses.dataclass(frozen=True, slots=True)
class Document:
    """Simple document with text content and arbitrary metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class StreamingProgress(TypedDict):
    """Progress info yielded during streaming insert."""

    batch_num: int
    total_batches: int | None  # None if unknown (infinite iterator)
    docs_processed: int
    docs_in_batch: int
    batch_ids: list[int]


# Type alias for progress callback
ProgressCallback = Callable[[StreamingProgress], None]


class DistanceStrategy(StrEnum):
    """Supported distance metrics for usearch backend."""

    COSINE = "cosine"
    L2 = "l2"  # euclidean (squared L2 internally)
    # Note: L1 (manhattan) was removed in v2.0.0 - usearch doesn't support it


class Quantization(StrEnum):
    FLOAT = "float"
    FLOAT16 = "float16"  # Half-precision: 2x memory savings, 1.5x speed
    INT8 = "int8"
    BIT = "bit"


class MigrationRequiredError(Exception):
    """Raised when a v1.x database needs migration to v2.0 usearch backend.

    This error is raised when opening a database that contains sqlite-vec
    data that needs to be migrated to the usearch backend.

    Attributes:
        path: Path to the database file
        collections: List of collection names that need migration
        total_vectors: Total number of vectors to migrate
        migration_info: Full migration info dict from check_migration()
    """

    def __init__(
        self,
        path: str,
        collections: list[str],
        total_vectors: int,
        migration_info: dict,
    ):
        self.path = path
        self.collections = collections
        self.total_vectors = total_vectors
        self.migration_info = migration_info

        msg = (
            f"Database '{path}' requires migration from sqlite-vec to usearch.\n"
            f"Collections: {', '.join(collections)} ({total_vectors} vectors total)\n\n"
            f"To migrate automatically, open with: VectorDB('{path}', auto_migrate=True)\n"
            f"Or check migration details first: VectorDB.check_migration('{path}')\n\n"
            f"⚠️  BACKUP YOUR DATABASE BEFORE MIGRATING: cp {path} {path}.backup"
        )
        super().__init__(msg)
