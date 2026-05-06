# src/simplevecdb/integrations/llamaindex.py
import logging
import sqlite3
import uuid
import warnings
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence

try:
    from llama_index.core.vector_stores import (
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.schema import TextNode, BaseNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        MetadataFilters,
        VectorStoreQueryMode,
    )
except ImportError as exc:
    raise ImportError(
        "LlamaIndex packages are no longer included by default. "
        "As of v2.3.0, install the integrations extra:\n\n"
        "  pip install simplevecdb[integrations]"
    ) from exc

_logger = logging.getLogger("simplevecdb.integrations.llamaindex")

from simplevecdb.core import VectorDB  # noqa: E402  (depends on llama_index probe above)

if TYPE_CHECKING:
    from simplevecdb.types import Document


class SimpleVecDBLlamaStore(BasePydanticVectorStore):
    """LlamaIndex-compatible wrapper for SimpleVecDB."""

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(
        self,
        db_path: str = ":memory:",
        collection_name: str = "default",
        **kwargs: Any,
    ):
        # Pass stores_text as a literal value, not self.stores_text
        super().__init__(stores_text=True)
        self._db = VectorDB(path=db_path, **kwargs)
        self._collection = self._db.collection(collection_name)
        # Map internal DB IDs to node IDs
        self._id_map: dict[int, str] = {}
        # Detect a v2.5 (or earlier) collection where the LlamaIndex node_id
        # was not persisted into metadata. delete(ref_doc_id) silently fails
        # against such rows because the metadata-fallback query finds
        # nothing. Emit a one-shot DeprecationWarning so the operator knows
        # to call migrate_node_id_metadata() before relying on delete().
        try:
            self._warn_if_legacy_collection()
        except Exception:
            _logger.debug(
                "Legacy-collection probe failed", exc_info=True
            )

    def _warn_if_legacy_collection(self) -> None:
        """Probe one row; warn if it lacks the v2.6 node_id metadata stamp."""
        sample = self._collection.get_documents(limit=1)
        if not sample:
            return
        _doc_id, _text, metadata = sample[0]
        if "_simplevecdb_node_id" in (metadata or {}):
            return
        warnings.warn(
            "SimpleVecDBLlamaStore: the underlying collection contains "
            "documents created before v2.6.0 that do not carry a "
            "'_simplevecdb_node_id' metadata stamp. delete(node_id) will "
            "silently no-op against those rows until you call "
            "store.migrate_node_id_metadata(). NOTE: legacy rows were "
            "stamped with the integer DB row id (not the original "
            "LlamaIndex node_id, which was never persisted), so "
            "delete(original_node_id) cannot be recovered for them; "
            "re-index to obtain stable node ids.",
            DeprecationWarning,
            stacklevel=3,
        )

    @property
    def client(self) -> Any:
        """Return the underlying client (our VectorDB)."""
        return self._db

    def migrate_node_id_metadata(self) -> int:
        """Backfill ``_simplevecdb_node_id`` for documents inserted before 2.6.0.

        Pre-2.6.0 versions did not persist the LlamaIndex node_id into
        document metadata, so ``delete(ref_doc_id)`` could not find the
        right row after a process restart. This helper walks every
        document in the underlying collection and stamps the internal DB
        id as the node_id for any row that lacks ``_simplevecdb_node_id``
        metadata. Idempotent — already-stamped rows are skipped, so a
        retry after a partial run converges.

        Limitation
        ----------
        Pre-2.6.0 rows never had their original LlamaIndex node_id
        persisted, so the backfill stamps ``str(doc_id)`` (the integer DB
        rowid) as the node_id. After migration:

        - ``delete(str(doc_id))`` works against migrated rows.
        - ``delete(<original-llama-uuid>)`` still cannot find these rows;
          the original ids were never written to disk and cannot be
          recovered. Re-indexing the upstream documents is the only way
          to restore stable node ids that match the LlamaIndex side.

        Atomicity: the underlying ``update_metadata`` issues one
        transaction per chunk of 500 rows. A process kill between chunks
        leaves the migration half-done; calling this method again will
        finish it (the per-row idempotency guard skips already-stamped
        rows).

        Returns:
            Number of documents updated.
        """
        docs = self._collection.get_documents()
        updates: list[tuple[int, dict[str, Any]]] = []
        for doc_id, _text, metadata in docs:
            if not metadata.get("_simplevecdb_node_id"):
                merged = dict(metadata or {})
                merged["_simplevecdb_node_id"] = str(doc_id)
                updates.append((int(doc_id), merged))
                self._id_map[int(doc_id)] = str(doc_id)
        if not updates:
            return 0
        return self._collection.update_metadata(updates)

    @property
    def store_text(self) -> bool:
        """Whether the store keeps text content."""
        return self.stores_text

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        """
        Add nodes with embeddings.

        The node_id is persisted into the document's metadata under
        ``_simplevecdb_node_id`` so it survives process restarts. The
        in-memory ``_id_map`` is also populated as a fast cache for the
        current session.
        """
        texts = [node.get_content() for node in nodes]

        # Stamp the node_id into metadata so delete() can recover the mapping
        # after a restart. When LlamaIndex did not assign a node_id, generate
        # a UUID up front so the metadata stamp is committed in the same
        # transaction as the row insert — there is no window where the row
        # exists without ``_simplevecdb_node_id`` in its metadata.
        node_ids_resolved: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for node in nodes:
            node_id = node.node_id or str(uuid.uuid4())
            md = dict(node.metadata or {})
            md["_simplevecdb_node_id"] = node_id
            node_ids_resolved.append(node_id)
            metadatas.append(md)

        embeddings = None
        if nodes and nodes[0].embedding is not None:
            emb_list = []
            all_have_embeddings = True
            for node in nodes:
                if node.embedding is None:
                    all_have_embeddings = False
                    break
                emb_list.append(node.embedding)
            if all_have_embeddings:
                embeddings = emb_list

        internal_ids = self._collection.add_texts(texts, metadatas, embeddings)

        for internal_id, node_id in zip(internal_ids, node_ids_resolved):
            self._id_map[internal_id] = node_id

        return node_ids_resolved

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete by ref_doc_id (node ID).

        First consults the in-memory ``_id_map`` for the current session;
        on a miss (typically after a restart) falls back to a metadata
        query against ``_simplevecdb_node_id`` so deletion is reliable
        across process boundaries.
        """
        internal_id: int | None = None
        for int_id, node_id in self._id_map.items():
            if node_id == ref_doc_id:
                internal_id = int_id
                break

        if internal_id is None:
            # Fall back to metadata lookup — mapping was not in this
            # process's _id_map, so we have to find it on disk. We catch
            # only TypeError (older catalog signatures lack
            # ``filter_dict=``) and NotImplementedError (filter mode not
            # supported); other exceptions — including database errors,
            # locked-DB, schema mismatches — propagate so the caller
            # cannot mistake a real failure for a successful no-op.
            try:
                docs = self._collection.get_documents(
                    filter_dict={"_simplevecdb_node_id": ref_doc_id}, limit=1
                )
            except (TypeError, NotImplementedError) as exc:
                _logger.debug(
                    "filter_dict-based delete fallback unavailable: %s",
                    exc,
                )
                docs = []
            except sqlite3.DatabaseError:
                # Database-layer error: surface it instead of swallowing.
                # A locked or corrupted DB previously turned into a
                # silent no-op, hiding data-loss bugs.
                raise
            if docs:
                internal_id = int(docs[0][0])

        if internal_id is not None:
            self._collection.delete_by_ids([internal_id])
            self._id_map.pop(internal_id, None)

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """
        Delete nodes from vector store.

        Args:
            node_ids: List of node IDs to delete.
            filters: Metadata filters. Currently unsupported — passing a
                non-None ``filters`` raises ``NotImplementedError`` rather
                than silently ignoring it (which would let callers think
                the deletion happened).
            **delete_kwargs: Unused.
        """
        if filters is not None:
            raise NotImplementedError(
                "delete_nodes(filters=...) is not yet supported by simplevecdb. "
                "Resolve the filter to node_ids first via the underlying "
                "VectorCollection.find_ids_by_filter() or query()."
            )
        if node_ids:
            for node_id in node_ids:
                self.delete(node_id)

    def _filters_to_dict(
        self, filters: MetadataFilters | None
    ) -> dict[str, Any] | None:
        if filters is None:
            return None
        result: dict[str, Any] = {}
        if hasattr(filters, "filters"):
            for filter_item in filters.filters:  # type: ignore[attr-defined]
                if hasattr(filter_item, "key") and hasattr(filter_item, "value"):
                    key = getattr(filter_item, "key")
                    value = getattr(filter_item, "value")
                    result[key] = value
        return result or None

    def _build_query_result(
        self,
        docs_with_scores: list[tuple["Document", float]],
        score_transform,
    ) -> VectorStoreQueryResult:
        nodes: list[TextNode] = []
        similarities: list[float] = []
        ids: list[str] = []

        for tiny_doc, score in docs_with_scores:
            # Prefer the persisted node_id over an unstable Python hash().
            # Python's hash() is randomized per process (PYTHONHASHSEED) and
            # can collide; ``_simplevecdb_node_id`` is stamped into metadata
            # at insert time, survives restarts, and uniquely identifies the
            # node.
            metadata = tiny_doc.metadata or {}
            node_id = metadata.get("_simplevecdb_node_id") or str(
                abs(hash(tiny_doc.page_content))
            )
            node = TextNode(
                text=tiny_doc.page_content,
                metadata=tiny_doc.metadata or {},
                id_=node_id,
                relationships={},
            )
            nodes.append(node)
            similarities.append(score_transform(score))
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Support dense, keyword, or hybrid lookups based on the requested mode."""

        filter_dict = self._filters_to_dict(query.filters)
        mode = getattr(query, "mode", VectorStoreQueryMode.DEFAULT)
        mode_value = getattr(mode, "value", mode)
        normalized_mode = str(mode_value).lower() if mode_value else "default"

        keyword_modes = {
            VectorStoreQueryMode.SPARSE.value,
            VectorStoreQueryMode.TEXT_SEARCH.value,
        }
        hybrid_modes = {
            VectorStoreQueryMode.HYBRID.value,
            VectorStoreQueryMode.SEMANTIC_HYBRID.value,
        }

        if normalized_mode in keyword_modes:
            if not query.query_str:
                raise ValueError("Keyword search requires query_str")
            results = self._collection.keyword_search(
                query.query_str,
                k=query.similarity_top_k,
                filter=filter_dict,
            )
            return self._build_query_result(results, lambda score: 1.0 / (1.0 + score))

        if normalized_mode in hybrid_modes:
            if not query.query_str:
                raise ValueError("Hybrid search requires query_str")
            results = self._collection.hybrid_search(
                query.query_str,
                k=query.similarity_top_k,
                filter=filter_dict,
                query_vector=query.query_embedding,
            )
            return self._build_query_result(results, lambda score: float(score))

        # Fallback to dense/vector search
        query_emb = query.query_embedding
        if query_emb is None:
            if query.query_str:
                query_input: str | list[float] = query.query_str
            else:
                raise ValueError("Either query_embedding or query_str must be provided")
        else:
            query_input = query_emb

        results = self._collection.similarity_search(
            query=query_input,
            k=query.similarity_top_k,
            filter=filter_dict,
        )
        return self._build_query_result(results, lambda distance: 1 - distance)
