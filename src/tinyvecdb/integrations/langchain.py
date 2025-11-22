from collections.abc import Iterable
from typing import Any

from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document as LangChainDocument

from tinyvecdb.core import VectorDB  # core class


class TinyVecDBVectorStore(VectorStore):
    """LangChain-compatible wrapper for TinyVecDB."""

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding: Embeddings | None = None,
        **kwargs: Any,
    ):
        self.embedding = embedding  # LangChain expects this
        self._db = VectorDB(path=db_path, **kwargs)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        db_path: str = ":memory:",
        **kwargs: Any,
    ) -> "TinyVecDBVectorStore":
        """
        Initialize from texts (embeds them automatically).

        Args:
            texts: List of texts to add.
            embedding: LangChain Embeddings model.
            metadatas: Optional list of metadata dicts.
            db_path: Path to SQLite database.
            **kwargs: Additional arguments for VectorDB.

        Returns:
            Initialized TinyVecDBVectorStore.
        """
        store = cls(embedding=embedding, db_path=db_path, **kwargs)
        store.add_texts(texts, metadatas)
        return store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Add texts (embed if no pre-computed). Returns IDs as str.

        Args:
            texts: Iterable of texts to add.
            metadatas: Optional list of metadata dicts.
            **kwargs: Additional arguments (e.g., ids).

        Returns:
            List of document IDs.
        """
        texts_list = list(texts)
        embeddings = None
        if self.embedding:
            embeddings = self.embedding.embed_documents(texts_list)
        ids = self._db.add_texts(
            texts=texts_list,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=kwargs.get("ids"),
        )
        return [str(id_) for id_ in ids]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[LangChainDocument]:
        """
        Search by text query (auto-embeds).

        Args:
            query: Text query string.
            k: Number of results to return.
            **kwargs: Additional arguments (e.g., filter).

        Returns:
            List of LangChain Documents.
        """
        if self.embedding:
            query_vec = self.embedding.embed_query(query)
        else:
            raise ValueError("Embedding model required for text queries")
        results = self._db.similarity_search(
            query=query_vec,
            k=k,
            filter=kwargs.get("filter"),
        )
        return [
            LangChainDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc, _ in results
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[LangChainDocument, float]]:
        """
        Return with scores (distances).

        Args:
            query: Text query string.
            k: Number of results to return.
            **kwargs: Additional arguments (e.g., filter).

        Returns:
            List of (Document, score) tuples.
        """
        if self.embedding:
            query_vec = self.embedding.embed_query(query)
        else:
            raise ValueError("Embedding model required")
        results = self._db.similarity_search(
            query=query_vec,
            k=k,
            filter=kwargs.get("filter"),
        )
        return [
            (
                LangChainDocument(page_content=doc.page_content, metadata=doc.metadata),
                score,
            )
            for doc, score in results
        ]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete.
            **kwargs: Unused.
        """
        if ids:
            int_ids = [int(id_) for id_ in ids]
            self._db.delete_by_ids(int_ids)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[LangChainDocument]:
        """
        Max marginal relevance search.

        Args:
            query: Text query string.
            k: Number of results to return.
            fetch_k: Number of candidates to fetch.
            lambda_mult: Diversity trade-off (unused in core currently).
            **kwargs: Additional arguments (e.g., filter).

        Returns:
            List of LangChain Documents.
        """
        if self.embedding:
            query_vec = self.embedding.embed_query(query)
        else:
            raise ValueError("Embedding model required for text queries")
        results = self._db.max_marginal_relevance_search(
            query=query_vec,
            k=k,
            fetch_k=fetch_k,
            filter=kwargs.get("filter"),
        )
        return [
            LangChainDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in results
        ]

    # Stub async (wrap sync for now – add true async in v1)
    async def aadd_texts(self, *args, **kwargs):
        return self.add_texts(*args, **kwargs)

    async def asimilarity_search(self, *args, **kwargs):
        return self.similarity_search(*args, **kwargs)

    # Other optional: max_marginal_relevance_search (implement via post-processing if needed)
    async def amax_marginal_relevance_search(
        self,
        *args,
        **kwargs,
    ) -> list[LangChainDocument]:
        return self.max_marginal_relevance_search(*args, **kwargs)
