from simplevecdb.types import Document, DistanceStrategy


def test_document_default_metadata_isolated():
    doc1 = Document(page_content="hello")
    doc2 = Document(page_content="world")

    assert doc1.metadata == {}
    assert doc2.metadata == {}
    assert doc1.metadata is not doc2.metadata


def test_distance_strategy_string_values():
    assert str(DistanceStrategy.COSINE) == "cosine"
    assert DistanceStrategy.L2.value == "l2"
    # Note: L1 removed in v2.0.0 (usearch doesn't support it)
