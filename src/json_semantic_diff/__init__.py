"""Semantic diff - structural similarity scoring for JSON documents."""

from __future__ import annotations

from json_semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig
from json_semantic_diff.api import (
    compare,
    compare_batch,
    compare_batch_pairs,
    consistency_score,
    format_diff,
    is_equivalent,
    similarity_score,
)
from json_semantic_diff.comparator import STEDComparator
from json_semantic_diff.result import ComparisonResult

__version__: str = "0.1.0"
__all__: list[str] = [
    "ArrayComparisonMode",
    "ComparisonResult",
    "STEDComparator",
    "STEDConfig",
    "compare",
    "compare_batch",
    "compare_batch_pairs",
    "consistency_score",
    "format_diff",
    "is_equivalent",
    "similarity_score",
]
