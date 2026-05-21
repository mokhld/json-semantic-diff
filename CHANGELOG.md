# Changelog

All notable changes to **json-semantic-diff** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The package is currently pre-1.0; public-API breaking changes may land in any `0.x` release.

## [Unreleased]

### Added
- CLI entry point `json-semantic-diff` for ad-hoc JSON comparison; flags `--json`, `--threshold`, `--verbose`, `--structural-weight`, `--content-weight`, `--array-mode`, `--version`; reads from stdin via `-`; 100 MiB hard cap on inputs (F8).
- `compare_batch(lefts, right, config=None)` and `compare_batch_pairs(pairs, config=None)` — share one `STEDComparator` so the embedding cache amortises across the batch (F5).
- `format_diff(result, indent=2)` — human-readable side-by-side renderer of a `ComparisonResult` (F6).
- `STEDConfig.ignore_paths` with `*` wildcard (F4); patterns are RFC 6901-aware so keys containing `/` or `~` can still be targeted.
- `STEDConfig.aliases` — bidirectional custom key-equivalence; short-circuits backend similarity to 1.0 (F7).
- `STEDConfig.numeric_tolerance` for float scalar comparison (I3).
- `STEDConfig.max_depth` for cost-capped traversal; identical sub-trees at the cap correctly score 0 cost (F3).
- `STEDConfig.collect_explanation` — when True, `ComparisonResult.explanation` carries a sorted `tuple[NodeContribution, ...]` of per-path attribution (F1).
- `ComparisonResult.to_dict()` and `to_json()` serialisers (D1).
- `NodeContribution` dataclass and `ComparisonResult.explanation` field, both re-exported from the top-level package.
- `PersistentEmbeddingCache` — diskcache-backed cache that survives process restart; install with `pip install json-semantic-diff[diskcache]` (F12).
- `FastEmbedBackend(cache_dir=..., local_files_only=...)` for offline / air-gapped use (F11).
- `OpenAIBackend(base_url=..., timeout=..., organization=...)` for Azure OpenAI, LiteLLM, vLLM, self-hosted endpoints (F10 partial).
- `EmbeddingCache(backend_id=...)` namespaces cache keys so model swaps are safe (H4).
- `[all]` aggregate optional extra installing every backend and integration (G4); new `[diskcache]` extra.
- Twelve Hypothesis property tests covering similarity bounds, idempotence, symmetry, threshold consistency, and serialisation round-trip (T8).
- Dedicated regression tests for NaN / inf scalars, unicode keys, mixed-type arrays, wide objects, bool/int conflation, array-mode symmetry, ignore_paths RFC 6901 escaping, and max_depth deep-identical handling.
- CI `test-extras` job that installs every optional backend and runs the full suite (CI1); new manual/weekly `benchmark` job runs the perf suite without slowing PR CI (CI2).
- `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`, `LICENSE` (D7, D6).
- `EmbeddingBackend` Protocol now requires both `embed()` AND `similarity()`; `StaticBackend`, `FastEmbedBackend`, `OpenAIBackend` all implement both. Custom backends without `similarity()` either need to add one or be wrapped in `EmbeddingCache`, which provides a cosine fallback (H8).

### Changed
- **Score-affecting:** `STEDConfig.lambda_unmatched` default bumped from `0.1` to `0.5` (audit I4). Wave 7 (C6) switched the `normalize_similarity` denominator to subtree-size sums; the previous `0.1` penalty contributed only ~5% of the size diff to the numerator, leaving "double the keys on one side" cases sitting near 0.7 instead of the intuitive ~0.5 floor. Identity (=1.0) and fully-disjoint type-mismatched (~0.0) are unchanged. Asymmetric structures drop measurably: `{"a":1}` vs `{"x":99,"y":"hello","z":[1,2]}` moves from 0.62 → 0.30; `{"a":{"b":1}}` vs `{"x":[1,2,3]}` from ~0.4 to ~0.0; the 45-pair Pearson correlation gate restores from > 0.80 to > 0.85. Same-shape pairs (size_diff = 0) are unaffected. Pass `lambda_unmatched=0.1` explicitly to recover the wave-7 behaviour.
- **Score-affecting:** OBJECT and ARRAY similarity now normalise by the proper Zhang-Shasha denominator (sum of children subtree sizes) instead of `len(children)` (audit C6). The previous denominator binary-collapsed the score when a matched-pair raw distance from a deep value subtree exceeded the parent's direct child count — a single deep leaf change in a wide-shallow document could score ~0. Under the new normaliser the same change scores ~0.99. Callers with pinned threshold values around ~0.5 will see the largest drift: same-shape, different-content OBJECTs (e.g. `{"name":"Alice"}` vs `{"price":99.99}`) move from ~0.0–0.1 up to ~0.5. The equivalence band (>= 0.85) and identity (=1.0) are unchanged.
- `ComparisonResult.matched_pairs`, `unmatched_left`, `unmatched_right` are now **tuples** (were lists). Iteration, indexing, and `len()` are unchanged; `.append()`, `.sort()`, and other mutating calls now raise. Convert with `list(result.matched_pairs)` if you need a mutable view. (D2)
- `tree/builder.py` emits RFC 6901-compliant JSON Pointer paths: object key segments escape `~` → `~0` and `/` → `~1` (tilde-first). Audit-trail paths in `ComparisonResult` change shape for keys containing `~` or `/`; ASCII keys are unaffected.
- `comparator._preprocess` always returns fresh structures regardless of `null_equals_missing`; user inputs are no longer mutated under any config (H10).
- `is_equivalent` validates `threshold ∈ [0.0, 1.0]` and raises `ValueError` otherwise; public-API docstrings now document `Raises:` sections (M3, M4).
- `EmbeddingCache.similarity()` cosine fallback clamps to `[0.0, 1.0]` (was unbounded below); identical floats no longer drift below zero, opposite vectors score 0.0 instead of -1.0. Only affects custom embed-only backends (H5).
- `OpenAIBackend.embed()` chunks inputs > 2048 client-side (H6); retry policy now covers `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError` with fallback to `APIError` on older SDKs (H7).
- `EmbeddingCache.embed()` dedupes inputs so duplicate strings never bill the wrapped backend twice (H3); now thread-safe via `threading.Lock` (P5).
- `BraintrustScorer` returns `BraintrustScore` (a `float` subclass carrying `.metadata` with key_mappings / unmatched paths / timing). `isinstance(x, float)` and arithmetic remain intact (F15, audit N4).
- `LangSmithEvaluator` embeds audit-trail JSON in `EvaluationResult.comment`; returns `score=None` (was 0.0) when `run.outputs` is missing or lacks the configured key (F15, audit N2).
- `WeaveScorer` score dict now carries `key_mappings`, `unmatched_left`, `unmatched_right`, `computation_time_ms`; returns `semantic_similarity=None` plus `skipped=True` (was 0.0) for missing reference (F15, audit N3).
- `backends/__init__.py` and `integrations/__init__.py` use static `__all__` + PEP 562 lazy `__getattr__` — type checkers and IDEs now see every optional symbol while runtime imports stay lazy (M5).
- `fastembed` pinned to `>=0.7,<1.0` to avoid the 0.3 → 0.7 API break; numerous wave-1 algorithm/backend/API correctness fixes from the 2026-05-21 audit (C1, C5, C6 partial, H2, H10, M2, M7, M10, M11, M12, P5, G3, F10 partial, D1, D2, D3, D6, D10, CI2, CI6, T10).

### Performance
- `EmbeddingCache.similarity()` now memoises pairwise scalar results (canonical (smaller-string-first) key, namespace-keyed by `backend_id`).  The algorithm's Hungarian cost-matrix pass and the comparator's downstream key-extraction pass share the cache, so the second pass becomes all-hits.  Wide-object benchmark: a 200-key 95%-disjoint compare drops from ~1.28 s to ~0.77 s on dev hardware (~40% faster); scores are bit-identical (I2).
- `scipy.optimize.linear_sum_assignment` is now imported lazily inside `hungarian_match()` rather than at module load.  Static-only consumers (no key matching past identity) no longer pay scipy's import cost; `import json_semantic_diff` drops by ~10–30 ms on cold caches (P3).

### Fixed
- AUTO array-mode symmetry property test (audit H9): the `@pytest.mark.xfail` marker was dropped — the test has been xpassing since wave 5 because the AUTO resolver inspects both arrays' contents in one pass, so the inferred ordered/unordered choice is invariant under argument swap.  Kept as a regression guard.
- Deep-recursion stack overflow (audit H1): `TreeBuilder` and `STEDAlgorithm._compute_node_distance` are now iterative explicit-stack walks. JSON nested past Python's default `sys.setrecursionlimit` no longer raises `RecursionError` from either tree construction or the STED traversal — a 10 000-deep linked-list-shaped document now compares cleanly. The public API is unchanged; the memoisation cache, depth threading, `max_depth` cap, explain-mode buffer and all scoring semantics are preserved bit-for-bit.
- `ignore_paths` patterns now correctly match keys containing `/` or `~` — the preprocessing layer now RFC 6901-escapes path segments before testing the pattern, matching the canonical paths emitted by the tree builder (review finding Adv5).
- `STEDConfig.max_depth` cap no longer scores deep-identical sub-trees as fully different. A shallow structural-equality check at the cap returns cost 0 when both sides are identical and the documented "declined comparison" cost otherwise (review finding M3 + Adv4).
- Bool / int conflation: `{flag: True}` no longer scores identical to `{flag: 1}` (H2).
- Tree builder rejects non-string dict keys (`TypeError`) and detects cyclic structures (`ValueError`) at the boundary instead of crashing deep in the algorithm (M11, M12).
- `tree/normalizer.py` regexes use `re.ASCII` so unicode keys (CJK, combining diacritics, emoji) are treated as single tokens rather than being split on accent boundaries (M10).
- Release workflow: tag push no longer uses `--force`; `github-release` job requires `publish-pypi` to actually succeed (was creating phantom releases on testpypi-only runs); `LICENSE` file added (C2–C4, D6).
- Stale `dist/semantic_diff-*` artefacts from the old package name removed so the next `poetry build` ships only `json_semantic_diff-*`.

### Removed
- `tests/test_placeholder.py` (no-op duplicate of `tests/packaging/test_packaging.py::test_version`).
- Module-level `_DEFAULT_BACKEND = FastEmbedBackend()` singleton in `backends/fastembed.py` — was unused and downloaded the ONNX model on import (C1).

### Known issues
- **PersistentEmbeddingCache trust requirement:** `diskcache` deserialises via `pickle`. `cache_dir` MUST be writable only by trusted principals; do not share across mutually-untrusting tenants. See `SECURITY.md`.
- C6 (full Zhang-Shasha denominator), H9 (AUTO array-mode symmetry), I2 (Hungarian-phase work amortised across passes via the pairwise cache), I4 (`lambda_unmatched` re-calibrated), and P3 (lazy scipy import) all landed in wave 8.  Remaining audit follow-ups: F2 (per-NodeType weights), F9/F13 (async API), F14 (cross-backend calibration), T11 (real-SDK adapter integration tests).  See `.planning/AUDIT-2026-05-21.md`.

## [0.0.x] — pre-history

Earlier work (project scaffolding, tree layer, STED algorithm with Hungarian matching and static backend, public API + comparator + embedding cache + consistency scorer, FastEmbed/OpenAI backends, evaluation-platform integrations, 661-test suite, CI + release workflows, rename to `json-semantic-diff`) landed before this changelog was started. See `git log` for full history.

[Unreleased]: https://github.com/mokhld/json-semantic-diff/compare/HEAD~1...HEAD
