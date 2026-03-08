# src/simplevecdb/integrations/llamaindex.py
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

from simplevecdb.core import VectorDB  # our core

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

    @property
    def client(self) -> Any:
        """Return the underlying client (our VectorDB)."""
        return self._db

    @property
    def store_text(self) -> bool:
        """Whether the store keeps text content."""
        return self.stores_text

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        """
        Add nodes with embeddings.

        Args:
            nodes: Sequence of LlamaIndex BaseNodes.
            **kwargs: Unused.

        Returns:
            List of node IDs.
        """
        texts = [node.get_content() for node in nodes]
        metadatas = [node.metadata for node in nodes]

        # Extract embeddings, ensuring all are valid or set to None
        embeddings = None
        if nodes and nodes[0].embedding is not None:
            # Ensure all embeddings are present (not None)
            emb_list = []
            all_have_embeddings = True
            for node in nodes:
                if node.embedding is None:
                    all_have_embeddings = False
                    break
                emb_list.append(node.embedding)

            if all_have_embeddings:
                embeddings = emb_list

        # Add to DB and get internal IDs
        internal_ids = self._collection.add_texts(texts, metadatas, embeddings)

        # Track mapping from internal ID to node ID
        node_ids = []
        for i, node in enumerate(nodes):
            internal_id = internal_ids[i]
            node_id = node.node_id or str(internal_id)
            self._id_map[internal_id] = node_id
            node_ids.append(node_id)

        return node_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete by ref_doc_id (node ID).

        Args:
            ref_doc_id: The node ID to delete.
            **delete_kwargs: Unused.
        """
        # Find internal ID from node ID
        internal_id = None
        for int_id, node_id in self._id_map.items():
            if node_id == ref_doc_id:
                internal_id = int_id
                break

        if internal_id is not None:
            self._collection.delete_by_ids([internal_id])
            del self._id_map[internal_id]

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
            filters: Metadata filters (unused).
            **delete_kwargs: Unused.
        """
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
            node_id = str(hash(tiny_doc.page_content))
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
