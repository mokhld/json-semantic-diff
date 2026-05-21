"""Property-based tests for json_semantic_diff using Hypothesis.

Each property is expressed as a separate ``@given`` test so that a failure
isolates the specific invariant that broke.  Strategies are kept small
(``max_leaves=20``, ``max_size=5``) to keep individual runs fast — the goal
is many examples per property, not deep structures per example.

KNOWN CAVEATS
-------------
- ``is_equivalent`` uses ``>= threshold`` semantics (see api.py).  The
  threshold-consistency test asserts exactly that equality.
- Top-level inputs are constrained to ``dict`` / ``list`` for the audit-trail
  properties, since ``matched_pairs`` / ``unmatched_left`` are only populated
  for OBJECT/OBJECT pairs.
"""

from __future__ import annotations

import json
import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from json_semantic_diff import (
    ArrayComparisonMode,
    STEDConfig,
    compare,
    consistency_score,
    is_equivalent,
    similarity_score,
)

# ---------------------------------------------------------------------------
# JSON value strategies
# ---------------------------------------------------------------------------

json_atoms = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
)

json_values = st.recursive(
    json_atoms,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=5),
    ),
    max_leaves=20,
)

# Top-level inputs constrained to dict/list (containers) for the audit-trail
# style properties.  We use a recursive children strategy so the *interior*
# can still hold scalars.
json_containers = st.one_of(
    st.lists(json_values, max_size=5),
    st.dictionaries(st.text(min_size=1, max_size=10), json_values, max_size=5),
)

# Dictionaries specifically — needed for the audit-trail property that
# asserts ``len(matched_pairs) + len(unmatched_left) == len(a)`` at the top
# level (this invariant only holds for top-level OBJECT inputs).
json_dicts = st.dictionaries(st.text(min_size=1, max_size=10), json_values, max_size=5)


# ---------------------------------------------------------------------------
# 1. Bounds: similarity_score is always in [0, 1]
# ---------------------------------------------------------------------------


@given(a=json_containers, b=json_containers)
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_similarity_in_unit_interval(a: object, b: object) -> None:
    """For any two JSON containers, the similarity score is in [0, 1]."""
    score = compare(a, b).similarity_score
    assert 0.0 <= score <= 1.0, f"score {score} outside [0, 1] for {a!r} vs {b!r}"


# ---------------------------------------------------------------------------
# 2. Idempotence: compare(a, a) == 1.0
# ---------------------------------------------------------------------------


@given(a=json_containers)
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_idempotence_identical_inputs(a: object) -> None:
    """compare(a, a) should yield similarity_score == 1.0 for any JSON value."""
    score = compare(a, a).similarity_score
    assert score == pytest.approx(1.0, abs=1e-9), (
        f"compare(a, a) returned {score} != 1.0 for a={a!r}"
    )


# ---------------------------------------------------------------------------
# 3. Symmetry: compare(a, b) == compare(b, a) for ORDERED / UNORDERED modes.
# ---------------------------------------------------------------------------

# Tolerance is generous because Hungarian / DP solvers can break ties
# differently when the two argument orders are swapped; the *scores* should
# still match to within float-arithmetic noise.
_SYMMETRY_TOL = 1e-9


@given(a=json_containers, b=json_containers)
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_symmetry_ordered_mode(a: object, b: object) -> None:
    """ORDERED array mode: similarity_score(a, b) == similarity_score(b, a)."""
    cfg = STEDConfig(array_comparison_mode=ArrayComparisonMode.ORDERED)
    s_ab = compare(a, b, config=cfg).similarity_score
    s_ba = compare(b, a, config=cfg).similarity_score
    assert math.isclose(s_ab, s_ba, abs_tol=_SYMMETRY_TOL), (
        f"ORDERED symmetry violated: {s_ab} vs {s_ba} for a={a!r} b={b!r}"
    )


@given(a=json_containers, b=json_containers)
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_symmetry_unordered_mode(a: object, b: object) -> None:
    """UNORDERED array mode: similarity_score(a, b) == similarity_score(b, a)."""
    cfg = STEDConfig(array_comparison_mode=ArrayComparisonMode.UNORDERED)
    s_ab = compare(a, b, config=cfg).similarity_score
    s_ba = compare(b, a, config=cfg).similarity_score
    assert math.isclose(s_ab, s_ba, abs_tol=_SYMMETRY_TOL), (
        f"UNORDERED symmetry violated: {s_ab} vs {s_ba} for a={a!r} b={b!r}"
    )


@given(a=json_containers, b=json_containers)
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_symmetry_auto_mode(a: object, b: object) -> None:
    """AUTO array mode: similarity_score(a, b) == similarity_score(b, a).

    Audit H9 (wave 8): the xfail-but-xpassing marker is gone — the AUTO
    mode resolution inspects both arrays' contents (``arr_a.children +
    arr_b.children``) so the inferred ordered-vs-unordered choice is
    invariant under argument swap.  Wave 5's T6 deterministic suite
    confirmed symmetry across 45 parametrised shapes; this property
    test extends the coverage to 50 random shapes per CI run.  Kept
    here as a regression guard.
    """
    cfg = STEDConfig(array_comparison_mode=ArrayComparisonMode.AUTO)
    s_ab = compare(a, b, config=cfg).similarity_score
    s_ba = compare(b, a, config=cfg).similarity_score
    assert math.isclose(s_ab, s_ba, abs_tol=_SYMMETRY_TOL), (
        f"AUTO symmetry violated: {s_ab} vs {s_ba} for a={a!r} b={b!r}"
    )


# ---------------------------------------------------------------------------
# 4. Threshold consistency:
#    is_equivalent(a, b, t) == (similarity_score(a, b) >= t)
# ---------------------------------------------------------------------------


@given(
    a=json_containers,
    b=json_containers,
    t=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_threshold_consistency(a: object, b: object, t: float) -> None:
    """is_equivalent must agree with the direct similarity_score >= threshold check."""
    expected = similarity_score(a, b) >= t
    actual = is_equivalent(a, b, threshold=t)
    assert actual == expected, (
        f"is_equivalent({a!r}, {b!r}, threshold={t}) returned {actual}; "
        f"score >= threshold returned {expected}"
    )


# ---------------------------------------------------------------------------
# 5. Audit-trail consistency:
#    The audit trail must be self-consistent with the inputs.  Two robust
#    sub-invariants (avoid relying on JSON Pointer path *parsing*, since
#    the library does not RFC-6901-escape '/' or '~' in keys):
#
#    (a) When the right side is ``{}``, every top-level key in the left dict
#        must show up in ``unmatched_left`` (the keys cannot be matched
#        against anything), so ``len(unmatched_left) >= len(a)``.
#    (b) When ``a == b`` (same dict), the top-level keys must all match —
#        ``unmatched_left`` and ``unmatched_right`` are empty, and the number
#        of matched_pairs is at least ``len(a)``.
# ---------------------------------------------------------------------------


@given(a=json_dicts)
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_audit_trail_unmatched_left_covers_all_keys_vs_empty(
    a: dict[str, object],
) -> None:
    """compare(a, {}) leaves every top-level key of ``a`` unmatched on the left."""
    result = compare(a, {})
    # No keys can be matched against an empty right side.
    assert result.matched_pairs == (), (
        f"matched_pairs should be empty when right={{}} got {result.matched_pairs!r}"
    )
    # Every top-level key of ``a`` must contribute at least one entry to
    # ``unmatched_left``.  Nested keys cannot appear (since matching halts at
    # the top OBJECT pair when the right side is empty), so the count is exact.
    assert len(result.unmatched_left) == len(a), (
        f"unmatched_left has {len(result.unmatched_left)} entries for a={a!r} "
        f"with {len(a)} top-level keys: {result.unmatched_left!r}"
    )
    assert result.unmatched_right == (), (
        f"unmatched_right should be empty when right={{}} got {result.unmatched_right!r}"
    )


@given(a=json_dicts)
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_audit_trail_self_compare_matches_everything(
    a: dict[str, object],
) -> None:
    """compare(a, a) matches every top-level key; nothing is unmatched."""
    result = compare(a, a)
    assert result.unmatched_left == (), (
        f"self-compare leaked unmatched_left={result.unmatched_left!r} for a={a!r}"
    )
    assert result.unmatched_right == (), (
        f"self-compare leaked unmatched_right={result.unmatched_right!r} for a={a!r}"
    )
    # ``matched_pairs`` includes top-level KEY pairs plus any nested OBJECT/OBJECT
    # KEY pairs reached recursively, so the count is ``>= len(a)``.
    assert len(result.matched_pairs) >= len(a), (
        f"self-compare matched_pairs ({len(result.matched_pairs)}) < top-level "
        f"key count ({len(a)}) for a={a!r}: {result.matched_pairs!r}"
    )


# ---------------------------------------------------------------------------
# 6. Serialisation round-trip:
#    result.to_dict() and result.to_json() are JSON-compatible.
# ---------------------------------------------------------------------------


@given(a=json_containers, b=json_containers)
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_result_serialisation_round_trips(a: object, b: object) -> None:
    """result.to_dict() must be JSON-dumpable; result.to_json() must parse back."""
    result = compare(a, b)

    # to_dict must json-dump cleanly
    raw_dict = result.to_dict()
    encoded = json.dumps(raw_dict)
    assert isinstance(encoded, str)

    # to_json must round-trip via json.loads
    parsed = json.loads(result.to_json())
    assert isinstance(parsed, dict)

    # Sanity: the six declared fields are present in the dict form
    for field in (
        "similarity_score",
        "matched_pairs",
        "key_mappings",
        "unmatched_left",
        "unmatched_right",
        "computation_time_ms",
    ):
        assert field in parsed, f"to_json output missing field {field!r}: {parsed}"

    # similarity_score in the parsed dict should match the original
    assert parsed["similarity_score"] == pytest.approx(
        result.similarity_score, abs=1e-12
    )


# ---------------------------------------------------------------------------
# 7. consistency_score: bounded, identity-on-singleton-and-empty.
# ---------------------------------------------------------------------------


@given(docs=st.lists(json_containers, max_size=4))
@settings(max_examples=50, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_consistency_score_bounded(docs: list[object]) -> None:
    """consistency_score returns a value in [0, 1] for any list of JSON values."""
    score = consistency_score(docs)
    assert 0.0 <= score <= 1.0, (
        f"consistency_score returned {score} outside [0, 1] for docs={docs!r}"
    )


@given(x=json_containers)
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_consistency_score_singleton_is_one(x: object) -> None:
    """consistency_score([x]) == 1.0 for any single document."""
    score = consistency_score([x])
    assert score == pytest.approx(1.0, abs=1e-9), (
        f"consistency_score([x]) returned {score} for x={x!r}"
    )


def test_consistency_score_empty_is_one() -> None:
    """consistency_score([]) returns 1.0 by definition (vacuous consistency)."""
    assert consistency_score([]) == pytest.approx(1.0, abs=1e-9)
