# Changelog

All notable changes to **json-semantic-diff** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The package is currently pre-1.0; public-API breaking changes may land in any `0.x` release.

## [Unreleased]

### Added
- CLI entry point `json-semantic-diff` for ad-hoc JSON comparison (F8).
- `STEDConfig.ignore_paths` with `*` wildcard support (F4).
- `FastEmbedBackend` offline-mode controls: `cache_dir` and `local_files_only` (F11).
- `[all]` aggregate extra installing every optional backend and integration (G4).
- Twelve Hypothesis property tests covering similarity, audit-trail, and threshold invariants (T8).
- CI `test-extras` job that installs every optional backend and runs the full suite (CI1).
- `CONTRIBUTING.md`, `CHANGELOG.md`, and `SECURITY.md` (D7).

### Changed
- `tree/builder.py` now emits RFC 6901-compliant JSON Pointer paths: object key segments escape `~` → `~0` and `/` → `~1` (tilde-first). Paths in `ComparisonResult.unmatched_left/right` and `matched_pairs` for keys containing `~` or `/` change shape — existing ASCII keys are unaffected.
- `fastembed` dependency pinned to `>=0.7,<1.0` to avoid surprise breakage from a 1.0 release (G1).
- Numerous algorithm, backend, API, DX, and packaging fixes from wave 1 of the 2026-05-21 audit (C1–C6 partial, H2–H8, H10, M2–M4, M7, M10–M12, P5, G3, F10 partial, D1–D3, D6, D10, CI2, CI6, T10).

### Fixed
- RFC 6901 path-escape bug: object keys containing `/` or `~` no longer produce ambiguous audit-trail paths (surfaced by Hypothesis property tests; tracked alongside the wave-2 work).

### Removed
- _(nothing user-visible)_

### Known issues
- Deep recursion (H1): documents nested past ~1000 levels can hit Python's recursion limit during tree construction. An iterative tree-walk rewrite is planned but not in this release. See `SECURITY.md` for input-validation guidance.
- OBJECT/ARRAY normalization collapse (C6 full fix), AUTO array-mode symmetry (H9), and several algorithmic improvements (I1–I7) remain open; see `.planning/AUDIT-2026-05-21.md`.

## [0.0.x] — pre-history

Earlier work (project scaffolding, tree layer, STED algorithm with Hungarian matching and static backend, public API + comparator + embedding cache + consistency scorer, FastEmbed/OpenAI backends, evaluation-platform integrations, 661-test suite, CI + release workflows, rename to `json-semantic-diff`) landed before this changelog was started. See `git log` for full history.

[Unreleased]: https://github.com/mokhld/json-semantic-diff/compare/HEAD~1...HEAD
