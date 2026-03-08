from __future__ import annotations

import sys
from types import ModuleType

from simplevecdb.utils import _import_optional


def test_import_optional_returns_existing_module(monkeypatch):
    dummy = ModuleType("_dummy_mod")
    monkeypatch.setitem(sys.modules, "_dummy_mod", dummy)

    result = _import_optional("_dummy_mod")

    assert result is dummy


def test_import_optional_handles_none_placeholder(monkeypatch):
    monkeypatch.setitem(sys.modules, "_none_mod", None)

    result = _import_optional("_none_mod")

    assert result is None


def test_import_optional_imports_module(monkeypatch):
    # Temporarily remove json so we exercise the import path
    original = sys.modules.pop("json", None)
    try:
        result = _import_optional("json")
        import json as json_module  # noqa: PLC0415

        assert result is json_module
    finally:
        if original is not None:
            sys.modules["json"] = original


def test_import_optional_missing_module():
    assert _import_optional("__missing_mod_xyz__") is None
