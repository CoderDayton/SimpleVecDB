"""Integrations package for SimpleVecDB.

Requires the 'integrations' extra: pip install simplevecdb[integrations]
"""


def __getattr__(name: str):
    if name == "SimpleVecDBVectorStore":
        from .langchain import SimpleVecDBVectorStore

        return SimpleVecDBVectorStore
    if name == "SimpleVecDBLlamaStore":
        from .llamaindex import SimpleVecDBLlamaStore

        return SimpleVecDBLlamaStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimpleVecDBVectorStore",
    "SimpleVecDBLlamaStore",
]
