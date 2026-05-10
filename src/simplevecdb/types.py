from __future__ import annotations

import dataclasses
from dataclasses import field
from enum import Enum
from typing import Callable, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


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


@dataclasses.dataclass
class ClusterResult:
    """Result of a clustering operation."""

    labels: np.ndarray
    centroids: np.ndarray | None
    doc_ids: list[int]
    n_clusters: int
    algorithm: str
    inertia: float | None = None
    silhouette_score: float | None = None

    def get_cluster_doc_ids(self, cluster_id: int) -> list[int]:
        """Get document IDs belonging to a specific cluster."""
        return [
            doc_id
            for doc_id, label in zip(self.doc_ids, self.labels)
            if label == cluster_id
        ]

    def summary(self) -> dict[int, int]:
        """Return cluster_id -> member count mapping."""
        from collections import Counter

        return dict(Counter(int(label) for label in self.labels))

    def metrics(self) -> dict[str, float | None]:
        """Return clustering quality metrics."""
        return {
            "inertia": self.inertia,
            "silhouette_score": self.silhouette_score,
        }


ClusterTagCallback = Callable[[list[str]], str]


@dataclasses.dataclass(frozen=True, slots=True)
class Edge:
    """Weighted directed edge between two documents (gap 3).

    Attributes:
        src_id: Source document id.
        dst_id: Destination document id.
        kind: Optional edge type label (default ""). Same (src, dst, kind)
            triple is unique; distinct kinds coexist between the same pair.
        weight: Numeric weight (e.g. similarity, plasticity).
        bonus: Secondary weight (free for caller — e.g. priority bias).
        hits: Counter of edge traversals/reinforcements.
        last_touch: Unix timestamp (seconds) of the most recent write.
        metadata: Optional extra JSON metadata.
    """

    src_id: int
    dst_id: int
    kind: str = ""
    weight: float = 0.0
    bonus: float = 0.0
    hits: int = 0
    last_touch: float = 0.0
    metadata: dict | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    """Change-feed entry (gap 7).

    Attributes:
        seq: Monotonic sequence number assigned by SQLite.
        ts: Unix timestamp (seconds) at append time.
        kind: Event kind (insert/update/delete/edge/counter/ttl/flush/...).
        doc_id: Optional document id this event refers to.
        payload: Optional JSON-decoded payload dict.
    """

    seq: int
    ts: float
    kind: str
    doc_id: int | None = None
    payload: dict | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class TTLEntry:
    """A pending expiry hook (gap 8).

    Attributes:
        doc_id: Target document id.
        expires_at: Unix timestamp (seconds) at which the entry expires.
        on_expire: Action when the entry expires ("delete" or "callback").
    """

    doc_id: int
    expires_at: float
    on_expire: str = "delete"
