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

### Fixed
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
- **Deep recursion (H1):** documents nested past ~1000 levels can hit Python's recursion limit during tree construction. An iterative tree-walk rewrite is planned. CLI catches `RecursionError` and exits 2 with a clean message; library users get no such guard. See `SECURITY.md` for input-validation guidance.
- **PersistentEmbeddingCache trust requirement:** `diskcache` deserialises via `pickle`. `cache_dir` MUST be writable only by trusted principals; do not share across mutually-untrusting tenants. See `SECURITY.md`.
- OBJECT/ARRAY normalization collapse (C6 full fix), AUTO array-mode symmetry (H9), Hungarian sparsity pre-filter (I2), and `lambda_unmatched` scaling (I4) remain open; see `.planning/AUDIT-2026-05-21.md`.

## [0.0.x] — pre-history

Earlier work (project scaffolding, tree layer, STED algorithm with Hungarian matching and static backend, public API + comparator + embedding cache + consistency scorer, FastEmbed/OpenAI backends, evaluation-platform integrations, 661-test suite, CI + release workflows, rename to `json-semantic-diff`) landed before this changelog was started. See `git log` for full history.

[Unreleased]: https://github.com/mokhld/json-semantic-diff/compare/HEAD~1...HEAD
