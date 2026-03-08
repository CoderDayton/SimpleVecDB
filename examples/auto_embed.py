"""
Auto-embedding example.

Requires: pip install "simplevecdb[server]"

This example shows how SimpleVecDB can auto-embed text if no embeddings
are provided. It uses local HuggingFace models via sentence-transformers.
"""

from simplevecdb import VectorDB

db = VectorDB(":memory:")
collection = db.collection("default")

try:
    # Auto-embeds using local model (requires [server] extras)
    collection.add_texts(["Paris is beautiful", "Berlin has great beer"])
    results = collection.similarity_search("Where should I drink beer?", k=1)
    print(results[0][0].page_content)  # → Berlin has great beer
except ValueError as e:
    if "No embeddings provided" in str(e):
        print("Auto-embedding requires [server] extras:")
        print("  pip install 'simplevecdb[server]'")
        print("\nAlternatively, provide embeddings explicitly:")
        print("  collection.add_texts(texts, embeddings=my_embeddings)")
    else:
        raise
