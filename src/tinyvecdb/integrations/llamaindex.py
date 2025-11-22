# src/tinyvecdb/integrations/llamaindex.py
from typing import Any
from collections.abc import Sequence

from llama_index.core.vector_stores import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore, MetadataFilters

from tinyvecdb.core import VectorDB  # our core


class TinyVecDBLlamaStore(BasePydanticVectorStore):
    """LlamaIndex-compatible wrapper for TinyVecDB."""

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(self, db_path: str = ":memory:", **kwargs: Any):
        # Pass stores_text as a literal value, not self.stores_text
        super().__init__(stores_text=True)
        self._db = VectorDB(path=db_path, **kwargs)
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
        internal_ids = self._db.add_texts(texts, metadatas, embeddings)

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
            self._db.delete_by_ids([internal_id])
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

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query with embedding or str (auto-embeds if str).

        Args:
            query: VectorStoreQuery object.
            **kwargs: Unused.

        Returns:
            VectorStoreQueryResult.
        """
        # Get query embedding - prefer embedding over string
        query_emb = query.query_embedding

        if query_emb is None:
            if query.query_str:
                # Try to use string query (requires embeddings in VectorDB)
                query_input: str | list[float] = query.query_str
            else:
                raise ValueError("Either query_embedding or query_str must be provided")
        else:
            query_input = query_emb

        # Convert MetadataFilters to simple dict if present
        filter_dict = None
        if query.filters is not None:
            # Convert LlamaIndex MetadataFilters to simple dict
            # For now, extract exact matches from filters
            filter_dict = {}
            if hasattr(query.filters, "filters"):
                for filter_item in query.filters.filters:
                    # Handle MetadataFilter objects with key and value attributes
                    # Use getattr with type ignore to handle dynamic attributes
                    if hasattr(filter_item, "key") and hasattr(filter_item, "value"):
                        key = getattr(filter_item, "key")
                        value = getattr(filter_item, "value")
                        filter_dict[key] = value

        # Perform search
        results = self._db.similarity_search(
            query=query_input,
            k=query.similarity_top_k,
            filter=filter_dict,
        )

        # Build response - nodes should be BaseNode objects, not NodeWithScore
        nodes = []
        similarities = []
        ids = []

        for tiny_doc, score in results:
            # Generate a stable node ID
            node_id = str(hash(tiny_doc.page_content))

            # Create TextNode with all required properties
            node = TextNode(
                text=tiny_doc.page_content,
                metadata=tiny_doc.metadata or {},
                id_=node_id,
                relationships={},  # Initialize empty relationships dict
            )

            # Convert distance to similarity (for cosine: similarity = 1 - distance)
            similarity = 1 - score

            # Return the node itself, not NodeWithScore - LlamaIndex will wrap it later
            nodes.append(node)
            similarities.append(similarity)
            ids.append(node_id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
