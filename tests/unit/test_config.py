"""Tests for config parsing helpers."""

from simplevecdb.config import _parse_registry, _parse_api_keys, _parse_bool_env


def test_parse_registry_empty():
    """Empty string returns default only."""
    result = _parse_registry(None, "default-model")
    assert result == {"default": "default-model"}


def test_parse_registry_alias_format():
    """Parse alias=repo pairs."""
    result = _parse_registry("local=model/repo,remote=other/repo", "default-model")
    assert result["local"] == "model/repo"
    assert result["remote"] == "other/repo"
    assert result["default"] == "default-model"


def test_parse_registry_bare_repo():
    """Bare repo IDs map to themselves."""
    result = _parse_registry("model/repo", "default-model")
    assert result["model/repo"] == "model/repo"
    assert result["default"] == "default-model"


def test_parse_registry_mixed():
    """Handle mix of alias=repo and bare entries."""
    result = _parse_registry("local=model/a,model/b", "default-model")
    assert result["local"] == "model/a"
    assert result["model/b"] == "model/b"


def test_parse_registry_whitespace():
    """Strip whitespace from entries."""
    result = _parse_registry("  local = model/repo  ,  model/b  ", "default-model")
    assert result["local"] == "model/repo"
    assert result["model/b"] == "model/b"


def test_parse_registry_empty_entries():
    """Skip empty entries between commas."""
    result = _parse_registry("local=model/repo,,model/b", "default-model")
    assert len(result) == 3  # local, model/b, default
    assert result["local"] == "model/repo"


def test_parse_api_keys_empty():
    """None or empty string returns empty set."""
    assert _parse_api_keys(None) == set()
    assert _parse_api_keys("") == set()


def test_parse_api_keys_single():
    """Single key."""
    result = _parse_api_keys("key123")
    assert result == {"key123"}


def test_parse_api_keys_multiple():
    """Multiple comma-separated keys."""
    result = _parse_api_keys("key1,key2,key3")
    assert result == {"key1", "key2", "key3"}


def test_parse_api_keys_whitespace():
    """Strip whitespace around keys."""
    result = _parse_api_keys("  key1  ,  key2  ")
    assert result == {"key1", "key2"}


def test_parse_api_keys_empty_entries():
    """Skip empty entries."""
    result = _parse_api_keys("key1,,key2,")
    assert result == {"key1", "key2"}


def test_parse_bool_env_none_uses_default():
    """None returns the default."""
    assert _parse_bool_env(None, True) is True
    assert _parse_bool_env(None, False) is False


def test_parse_bool_env_truthy():
    """Truthy strings return True."""
    assert _parse_bool_env("1", False) is True
    assert _parse_bool_env("true", False) is True
    assert _parse_bool_env("TRUE", False) is True
    assert _parse_bool_env("yes", False) is True
    assert _parse_bool_env("YES", False) is True
    assert _parse_bool_env("on", False) is True
    assert _parse_bool_env("anything", False) is True


def test_parse_bool_env_falsey():
    """Falsey strings return False."""
    assert _parse_bool_env("0", True) is False
    assert _parse_bool_env("false", True) is False
    assert _parse_bool_env("FALSE", True) is False
    assert _parse_bool_env("no", True) is False
    assert _parse_bool_env("NO", True) is False
    assert _parse_bool_env("off", True) is False
    assert _parse_bool_env("OFF", True) is False


def test_parse_bool_env_whitespace():
    """Handle whitespace in input."""
    assert _parse_bool_env("  0  ", True) is False
    assert _parse_bool_env("  1  ", False) is True
