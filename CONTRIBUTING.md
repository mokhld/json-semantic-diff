# Contributing to json-semantic-diff

Thanks for your interest. This project is alpha (Development Status :: 3 - Alpha); the public surface and internal layout are still moving. Small, focused PRs with tests are easiest to review.

## Local setup

Requires Python 3.11+ and [Poetry](https://python-poetry.org/) 2.x.

```bash
git clone https://github.com/mokhld/json-semantic-diff
cd json-semantic-diff
poetry install --with dev --extras all
poetry run pre-commit install
```

`--extras all` pulls every optional backend (FastEmbed, OpenAI) and every evaluation-platform integration (LangSmith, Braintrust, Weave) so the full test suite can run. If you only need core behaviour, drop the `--extras` flag.

## Running tests

```bash
poetry run pytest                       # default unit + integration suite
poetry run pytest tests/benchmarks --benchmark-only   # perf benchmarks (gated)
```

Benchmarks live under `tests/benchmarks/` and are excluded from default runs (see `tool.pytest.ini_options.norecursedirs` in `pyproject.toml`). Extras-dependent tests require `--extras all` to be installed; if you skipped it, those tests will fail to import.

Property tests (`tests/test_properties.py`) use Hypothesis. They run as part of the default suite — keep them fast (`max_examples=50`, `deadline=2000`).

## Code style

`ruff` and `mypy --strict` are both enforced via pre-commit and CI:

```bash
poetry run ruff check src tests
poetry run ruff format src tests
poetry run mypy --strict src/json_semantic_diff
```

Lint rules: `E, W, F, I, UP, B, SIM, PT, RUF`. Line length is 88. The mypy config sets `strict = true` plus `warn_return_any`, `warn_unused_ignores`, `warn_unreachable`. Tests are exempt from `disallow_untyped_defs`, but everything under `src/` must type-check cleanly.

## Adding a backend

A backend is any class that conforms to the `EmbeddingBackend` Protocol in `src/json_semantic_diff/protocols.py`:

```python
class EmbeddingBackend(Protocol):
    def embed(self, strings: list[str]) -> np.ndarray: ...
    def similarity(self, a: str, b: str) -> float: ...
```

The Protocol is `@runtime_checkable`, so no inheritance is required — duck typing is enough. See `src/json_semantic_diff/backends/static.py` (zero-dependency Levenshtein) and `src/json_semantic_diff/backends/fastembed.py` (local ONNX, with offline `cache_dir` / `local_files_only` support) for reference implementations. Wrap your backend in `json_semantic_diff.cache.EmbeddingCache` if you want LRU caching for free.

## Opening a PR

- Keep PRs small and focused on one concern.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `ci:` are the prefixes you'll see in `git log`).
- Behaviour changes require tests. Bug fixes should add a regression test that fails without the fix.
- CI runs ruff, mypy --strict, and the full pytest suite (including an extras job that installs every optional dependency). All three must pass.
- Update `CHANGELOG.md` under `## [Unreleased]` for user-visible changes.

## Reading the architecture

A 60-second tour of the package layout under `src/json_semantic_diff/`:

- `algorithm/sted.py` — the STED (Semantic Tree Edit Distance) core: Zhang-Shasha tree edit distance with Hungarian semantic key matching. `algorithm/config.py` and `algorithm/costs.py` carry the tuning knobs.
- `tree/` — JSON-to-tree parsing. `builder.py` turns Python values into typed `TreeNode` trees; `normalizer.py` handles naming-convention folding; `nodes.py` defines the node types.
- `backends/` — embedding/similarity backends (`static.py`, `fastembed.py`, `openai.py`). All implement the `EmbeddingBackend` Protocol from `protocols.py`.
- `comparator.py` — the `STEDComparator` that wires backend + algorithm + preprocessing.
- `api.py` — the public surface (`compare`, `similarity_score`, `is_equivalent`).
- `integrations/` — adapters for pytest, LangSmith, Braintrust, and Weave.
- `cache.py`, `scorer.py`, `result.py` — embedding cache, generator-consistency scorer, and the `ComparisonResult` dataclass.

For deeper internals, the audit notes in `.planning/AUDIT-2026-05-21.md` summarise known issues and recent work.
