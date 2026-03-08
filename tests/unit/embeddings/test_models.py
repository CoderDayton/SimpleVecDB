"""Tests for embeddings model loading and inference."""

from unittest.mock import MagicMock, patch
from pathlib import Path

from simplevecdb.embeddings.models import (
    _load_sentence_transformer_cls,
    _load_snapshot_download,
    load_model,
    get_embedder,
    embed_texts,
    _loaded_models,
)


def test_load_sentence_transformer_cls():
    """Test lazy import of SentenceTransformer class."""
    mock_st = MagicMock()
    with patch.dict(
        "sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}
    ):
        cls = _load_sentence_transformer_cls()
        assert cls is mock_st


def test_load_sentence_transformer_cls_missing_dep():
    """Test ImportError when sentence-transformers not installed."""
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            try:
                _load_sentence_transformer_cls()
                assert False, "Should raise ImportError"
            except ImportError as e:
                assert "simplevecdb[server]" in str(e)


def test_load_snapshot_download():
    """Test lazy import of snapshot_download."""
    mock_sd = MagicMock()
    with patch.dict(
        "sys.modules", {"huggingface_hub": MagicMock(snapshot_download=mock_sd)}
    ):
        fn = _load_snapshot_download()
        assert fn is mock_sd


def test_load_snapshot_download_missing_dep():
    """Test ImportError when huggingface_hub not installed."""
    with patch.dict("sys.modules", {"huggingface_hub": None}):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            try:
                _load_snapshot_download()
                assert False, "Should raise ImportError"
            except ImportError as e:
                assert "simplevecdb[server]" in str(e)


def test_load_model():
    """Test load_model downloads and initializes model."""
    mock_model = MagicMock()
    mock_st_cls = MagicMock(return_value=mock_model)
    mock_snapshot = MagicMock(return_value="/cache/model-path")

    with patch(
        "simplevecdb.embeddings.models._load_sentence_transformer_cls",
        return_value=mock_st_cls,
    ):
        with patch(
            "simplevecdb.embeddings.models._load_snapshot_download",
            return_value=mock_snapshot,
        ):
            with patch("simplevecdb.embeddings.models.CACHE_DIR", Path("/tmp/cache")):
                result = load_model("test/model")

    assert result is mock_model
    mock_snapshot.assert_called_once()
    assert mock_snapshot.call_args.kwargs["repo_id"] == "test/model"
    mock_st_cls.assert_called_once_with(
        "/cache/model-path",
        tokenizer_kwargs={"padding": True, "truncation": True, "max_length": 512},
        backend="torch",
    )


def test_get_embedder_caches_models():
    """Test get_embedder caches loaded models."""
    _loaded_models.clear()
    mock_model = MagicMock()

    with patch(
        "simplevecdb.embeddings.models.load_model", return_value=mock_model
    ) as mock_load:
        # First call loads
        result1 = get_embedder("test/model")
        assert result1 is mock_model
        mock_load.assert_called_once_with("test/model")

        # Second call uses cache
        result2 = get_embedder("test/model")
        assert result2 is mock_model
        assert mock_load.call_count == 1  # Not called again

    _loaded_models.clear()


def test_get_embedder_uses_default():
    """Test get_embedder uses DEFAULT_MODEL when model_id is None."""
    _loaded_models.clear()
    mock_model = MagicMock()

    with patch("simplevecdb.embeddings.models.load_model", return_value=mock_model):
        with patch("simplevecdb.embeddings.models.DEFAULT_MODEL", "default/model"):
            result = get_embedder(None)

    assert result is mock_model
    _loaded_models.clear()


def test_embed_texts_empty():
    """Test embed_texts with empty list."""
    result = embed_texts([])
    assert result == []


def test_embed_texts_calls_model():
    """Test embed_texts encodes texts with model."""
    mock_model = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_model.encode.return_value = mock_embeddings

    with patch("simplevecdb.embeddings.models.get_embedder", return_value=mock_model):
        with patch("simplevecdb.embeddings.models.config") as mock_config:
            mock_config.EMBEDDING_BATCH_SIZE = 32
            result = embed_texts(["hello", "world"], model_id="test/model")

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_model.encode.assert_called_once_with(
        ["hello", "world"],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )


def test_embed_texts_batch_size_override():
    """Test embed_texts respects batch_size parameter."""
    mock_model = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.tolist.return_value = [[0.1]]
    mock_model.encode.return_value = mock_embeddings

    with patch("simplevecdb.embeddings.models.get_embedder", return_value=mock_model):
        embed_texts(["test"], batch_size=128)

    assert mock_model.encode.call_args.kwargs["batch_size"] == 128


def test_embed_texts_uses_default_model():
    """Test embed_texts uses default model when model_id not specified."""
    _loaded_models.clear()
    mock_model = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.tolist.return_value = [[0.1]]
    mock_model.encode.return_value = mock_embeddings

    with patch(
        "simplevecdb.embeddings.models.get_embedder", return_value=mock_model
    ) as mock_get:
        with patch("simplevecdb.embeddings.models.DEFAULT_MODEL", "default/model"):
            embed_texts(["test"])

    mock_get.assert_called_once_with(None)
    _loaded_models.clear()
