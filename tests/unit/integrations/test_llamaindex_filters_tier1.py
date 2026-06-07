"""Tier-1 fix: LlamaIndex metadata filter operators/conditions are honored."""

from __future__ import annotations

import pytest

try:
    import llama_index  # noqa: F401
except ImportError:
    pytest.skip("llama-index not installed", allow_module_level=True)

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore


def _store() -> SimpleVecDBLlamaStore:
    return SimpleVecDBLlamaStore(db_path=":memory:")


def test_gt_operator_maps_to_dollar_gt() -> None:
    f = MetadataFilters(
        filters=[MetadataFilter(key="score", value=0.5, operator=FilterOperator.GT)]
    )
    assert _store()._filters_to_dict(f) == {"score": {"$gt": 0.5}}


def test_eq_operator_stays_bare_value() -> None:
    f = MetadataFilters(
        filters=[MetadataFilter(key="tag", value="x", operator=FilterOperator.EQ)]
    )
    assert _store()._filters_to_dict(f) == {"tag": "x"}


def test_or_condition_raises_not_implemented() -> None:
    f = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.EQ),
            MetadataFilter(key="b", value=2, operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR,
    )
    with pytest.raises(NotImplementedError):
        _store()._filters_to_dict(f)


def test_in_operator_maps_to_dollar_in() -> None:
    f = MetadataFilters(
        filters=[
            MetadataFilter(key="tag", value=["a", "b"], operator=FilterOperator.IN)
        ]
    )
    assert _store()._filters_to_dict(f) == {"tag": {"$in": ["a", "b"]}}


def test_not_condition_raises_not_implemented() -> None:
    f = MetadataFilters(
        filters=[
            MetadataFilter(key="a", value=1, operator=FilterOperator.EQ),
            MetadataFilter(key="b", value=2, operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.NOT,
    )
    with pytest.raises(NotImplementedError):
        _store()._filters_to_dict(f)
