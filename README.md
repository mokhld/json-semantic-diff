# json-semantic-diff

**Semantic similarity scoring for JSON structures — not just whether they differ, but *how similar* they are.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-661%20passed-brightgreen.svg)]()
[![Type Checked](https://img.shields.io/badge/mypy-strict-blue.svg)]()

## Overview

`json-semantic-diff` compares two JSON documents and returns a normalised similarity score in [0.0, 1.0], along with an audit trail of which keys matched, which were renamed, and which are missing. It handles naming convention differences (camelCase, snake_case, PascalCase, kebab-case) transparently.

Traditional JSON diff tools are binary: match or no match. `json-semantic-diff` tells you *how similar* two documents are, which is what you actually need when testing LLM outputs, validating API migrations, or measuring generator stability.

Core algorithm: **STED** (Semantic Tree Edit Distance) — tree edit distance extended with semantic key matching via the Hungarian algorithm and per-level normalisation.

## Key Features

- **Similarity scoring** — Normalised float in [0.0, 1.0] instead of binary match/no-match
- **Naming convention tolerance** — `user_name`, `userName`, `UserName`, `user-name` all score 1.0 against each other
- **Rich audit trail** — Key mappings, matched pairs, unmatched paths, computation time
- **Generator consistency** — Single-number stability metric for LLM outputs
- **Three backends** — Zero-dependency Levenshtein, local ONNX embeddings, or OpenAI cloud
- **Evaluation platform adapters** — pytest, LangSmith, Braintrust, W&B Weave
- **Configurable** — Structural/content weights, array comparison modes, type coercion, null handling
- **Zero global state** — Each comparator instance owns its own cache; thread-safe by design

## Installation

```bash
pip install json-semantic-diff
```

With optional backends:

```bash
pip install json-semantic-diff[fastembed]   # Local ONNX embeddings (384-dim)
pip install json-semantic-diff[openai]      # OpenAI cloud embeddings (1536-dim)
```

With evaluation platform adapters:

```bash
pip install json-semantic-diff[langsmith]
pip install json-semantic-diff[braintrust]
pip install json-semantic-diff[weave]
```

## Quick Start

### Compare two JSON documents

```python
from json_semantic_diff import compare

# Note: `mailAddress` vs `email_address` is a fuzzy (not exact) key match,
# so the score is close to but below 1.0.
result = compare(
    {"user_name": "Alice", "email_address": "alice@corp.com"},
    {"userName": "Alice", "mailAddress": "alice@corp.com"},
)

print(result.similarity_score)   # ~0.98
print(result.key_mappings)       # {"user_name": "userName", "email_address": "mailAddress"}
print(result.unmatched_left)     # ()
print(result.unmatched_right)    # ()
print(result.computation_time_ms)  # sub-millisecond on small docs

# Naming-convention-only differences score a perfect 1.0:
compare(
    {"user_name": "Alice"},
    {"userName": "Alice"},
).similarity_score  # 1.0
```

### Quick similarity score

```python
from json_semantic_diff import similarity_score

# Pure naming-convention difference -> identical after normalisation.
score = similarity_score(
    {"first_name": "Bob", "last_name": "Smith"},
    {"firstName": "Bob", "lastName": "Smith"},
)
print(score)  # 1.0
```

### Boolean equivalence check

```python
from json_semantic_diff import is_equivalent

# Passes — naming convention difference only
is_equivalent({"user_name": "Alice"}, {"userName": "Alice"})  # True

# Fails — different structure entirely
is_equivalent({"name": "Alice"}, {"product": "Widget"})  # False

# Custom threshold
is_equivalent({"user_name": "Alice"}, {"userName": "Alice"}, threshold=0.99)
```

### Measure generator consistency

```python
from json_semantic_diff import consistency_score

# Stable generator → 1.0
docs = [
    {"name": "Alice", "age": 30},
    {"name": "Alice", "age": 30},
    {"name": "Alice", "age": 30},
]
print(consistency_score(docs))  # 1.0

# Erratic generator → low score
erratic = [
    {"name": "Alice", "age": 30},
    {"fullName": "Alice", "years": 30},
    {"person": "Alice"},
]
print(consistency_score(erratic))  # ~0.15
```

## API Reference

### Public Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `compare(left, right, config=None)` | Two JSON values + optional config | `ComparisonResult` |
| `similarity_score(left, right, config=None)` | Two JSON values + optional config | `float` in [0.0, 1.0] |
| `is_equivalent(left, right, threshold=0.85, config=None)` | Two JSON values + threshold + config | `bool` |
| `consistency_score(docs, config=None)` | List of JSON values + optional config | `float` in [0.0, 1.0] |

### ComparisonResult

Returned by `compare()`. Frozen dataclass with six fields:

| Field | Type | Description |
|-------|------|-------------|
| `similarity_score` | `float` | Normalised similarity in [0.0, 1.0]. 1.0 = identical. |
| `matched_pairs` | `list[tuple[str, str]]` | JSON Pointer pairs for matched KEY nodes |
| `key_mappings` | `dict[str, str]` | Raw left key name → raw right key name |
| `unmatched_left` | `list[str]` | JSON Pointer paths with no match in right document |
| `unmatched_right` | `list[str]` | JSON Pointer paths with no match in left document |
| `computation_time_ms` | `float` | Wall-clock duration in milliseconds |

### STEDConfig

Immutable configuration for the algorithm. All parameters have sensible defaults.

```python
from json_semantic_diff import STEDConfig, ArrayComparisonMode

config = STEDConfig(
    w_s=0.5,                                          # Structural weight [0, 1]
    w_c=0.5,                                          # Content weight [0, 1] (must sum to 1.0)
    lambda_unmatched=0.1,                              # Penalty for unmatched children (>= 0)
    array_comparison_mode=ArrayComparisonMode.ORDERED,  # ORDERED | UNORDERED | AUTO
    type_coercion=False,                               # "42" == 42?
    null_equals_missing=False,                         # {x: null} == {}?
)

result = compare(doc1, doc2, config=config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `w_s` | `float` | `0.5` | Structural weight. Higher → structure matters more. |
| `w_c` | `float` | `0.5` | Content weight. Higher → values matter more. Must sum to 1.0 with `w_s`. |
| `lambda_unmatched` | `float` | `0.1` | Penalty per unmatched child. 0.0 = ignore extras. 1.0 = full penalty. |
| `array_comparison_mode` | `ArrayComparisonMode` | `ORDERED` | How arrays are compared. |
| `type_coercion` | `bool` | `False` | When True, `"42"` and `42` compare as equal. |
| `null_equals_missing` | `bool` | `False` | When True, `{"x": null}` and `{}` compare as equal. |

### Array Comparison Modes

| Mode | Strategy | Use case |
|------|----------|----------|
| `ORDERED` | Positional DP alignment | Sequences, logs, ordered lists |
| `UNORDERED` | Hungarian matching | Tags, feature flags, sets |
| `AUTO` | Infer from content (scalars → unordered, objects → ordered) | When array semantics vary |

## Backends

Three embedding backends for key matching, each optimising for different constraints:

| Backend | Dependencies | Latency | Key matching quality | Cost |
|---------|-------------|---------|---------------------|------|
| **StaticBackend** (default) | None | ~0.1ms | Naming conventions | Free |
| **FastEmbedBackend** | `fastembed` | ~1-2s cold start | Semantic understanding | Free |
| **OpenAIBackend** | `openai`, `tenacity` | ~100-500ms | Best quality | API costs |

### StaticBackend (default)

Uses Levenshtein edit distance on normalised keys. No ML model, no API calls.

```python
from json_semantic_diff.backends import StaticBackend

backend = StaticBackend()
backend.similarity("user_name", "userName")   # 1.0
backend.similarity("user_name", "address")    # < 0.5
```

### FastEmbedBackend

Local ONNX embeddings via `sentence-transformers/all-MiniLM-L6-v2` (384-dim).

```python
from json_semantic_diff.backends.fastembed import FastEmbedBackend
from json_semantic_diff.comparator import STEDComparator

backend = FastEmbedBackend()  # ~1-2s cold start for model loading
cmp = STEDComparator(backend=backend)
result = cmp.compare(doc1, doc2)
```

### OpenAIBackend

Cloud embeddings via `text-embedding-3-small` (1536-dim). API key from `OPENAI_API_KEY` environment variable.

```python
from json_semantic_diff.backends.openai import OpenAIBackend
from json_semantic_diff.comparator import STEDComparator

backend = OpenAIBackend()  # Reads OPENAI_API_KEY from env
cmp = STEDComparator(backend=backend)
result = cmp.compare(doc1, doc2)
```

Auto-retries rate-limited requests with jittered exponential backoff (up to 6 attempts via tenacity).

### Using STEDComparator directly

For batch comparisons where you want cache reuse across calls:

```python
from json_semantic_diff.comparator import STEDComparator

cmp = STEDComparator()  # Or with backend= and config=
for doc in documents:
    result = cmp.compare(doc, reference)
    # Embedding cache reused across all calls
```

## Integrations

### pytest

Auto-discovered fixture — install the package and it's available in every test:

```python
def test_api_response(assert_json_equivalent):
    actual = get_user_from_api()
    expected = {"user_name": "Alice", "age": 30}

    # Passes even if API returns {"userName": "Alice", "age": 30}
    assert_json_equivalent(actual, expected)
```

Custom threshold and config:

```python
from json_semantic_diff import STEDConfig

def test_strict(assert_json_equivalent):
    config = STEDConfig(type_coercion=True)
    assert_json_equivalent(actual, expected, threshold=0.95, config=config)
```

Failure messages include full context:

```
AssertionError: JSON documents not equivalent:
  similarity=0.42 < threshold=0.85
  actual:   {"product": "Widget", "price": 19.99}
  expected: {"user_name": "Alice", "age": 30}
  key_mappings: {"product": "user_name"}
  unmatched_left: ["/price"]
  unmatched_right: ["/age"]
```

### LangSmith

```python
from json_semantic_diff.comparator import STEDComparator
from json_semantic_diff.integrations import LangSmithEvaluator

comparator = STEDComparator()
evaluator = LangSmithEvaluator(comparator, output_key="response")
# Pass directly to langsmith.evaluate()
```

### Braintrust

```python
from json_semantic_diff.comparator import STEDComparator
from json_semantic_diff.integrations import BraintrustScorer

comparator = STEDComparator()
scorer = BraintrustScorer(comparator)
# Returns float in [0.0, 1.0] or None when no expected value
```

### W&B Weave

```python
from json_semantic_diff.comparator import STEDComparator
from json_semantic_diff.integrations import WeaveScorer

comparator = STEDComparator()
scorer = WeaveScorer(comparator)
# Returns {"semantic_similarity": float}
```

## How It Works

### The STED Algorithm

1. **JSON → Typed Tree.** Each JSON value becomes a tree node with a type (OBJECT, KEY, ARRAY, ELEMENT, SCALAR). KEY nodes store both raw and normalised labels.

2. **Key Normalisation.** A five-pass regex pipeline converts all naming conventions to a canonical form: `userName` → `"user name"`, `user_name` → `"user name"`, `UserName` → `"user name"`, `user-name` → `"user name"`.

3. **Hungarian Key Matching.** For each pair of OBJECT nodes, a cost matrix is built using key similarity scores. The Hungarian algorithm finds the optimal key-to-key assignment.

4. **Blended Cost.** Each node comparison blends structural and content distance: `cost = w_s * structural_distance + w_c * content_distance`.

5. **Per-Level Normalisation.** At each tree level: `similarity = 1.0 - min(1.0, [d_matched + λ·|n_left - n_right|] / max(n_left, n_right, 1))`. This keeps scores in [0.0, 1.0] regardless of document depth.

### Consistency Score Formula

```
pairwise_scores = [compare(docs[i], docs[j]).similarity_score for all unique (i, j) pairs]
consistency = max(0, mean(pairwise_scores) - std(pairwise_scores))
```

Penalises both low average similarity (different outputs) and high variance (erratic outputs).

### Embedding Cache

All unique KEY labels are collected before the algorithm runs and embedded in a single batch call. The algorithm then runs entirely from cache — zero additional backend calls per comparison.

Each `STEDComparator` instance owns its own cache (no global state). The public API functions create a fresh comparator per call for isolation. Use `STEDComparator` directly for batch comparisons where cache reuse matters.

## Development

```bash
git clone https://github.com/mokhld/json-semantic-diff.git
cd json-semantic-diff
poetry install --with dev
```

### Running tests

```bash
poetry run pytest                    # All 661 tests
poetry run pytest -x                 # Stop on first failure
poetry run pytest tests/unit/        # Unit tests only
poetry run pytest tests/algorithm/   # Algorithm tests only
```

### Type checking

```bash
poetry run mypy src/          # Strict mode
```

### Linting

```bash
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/
```

### Project structure

```
src/json_semantic_diff/
├── __init__.py              # Public API exports
├── api.py                   # 4 public functions
├── comparator.py            # STEDComparator orchestrator
├── scorer.py                # ConsistencyScorer
├── result.py                # ComparisonResult dataclass
├── protocols.py             # EmbeddingBackend Protocol
├── cache.py                 # EmbeddingCache (LRU)
├── algorithm/
│   ├── config.py            # STEDConfig + ArrayComparisonMode
│   ├── sted.py              # STEDAlgorithm (recursive core)
│   ├── costs.py             # Insert/delete/update cost functions
│   ├── matcher.py           # Hungarian algorithm wrapper
│   └── normalizer.py        # Per-level similarity normalisation
├── backends/
│   ├── static.py            # Levenshtein (zero dependencies)
│   ├── fastembed.py         # ONNX (local embeddings)
│   └── openai.py            # OpenAI (cloud embeddings)
├── tree/
│   ├── nodes.py             # TreeNode + NodeType
│   ├── builder.py           # JSON → Tree conversion
│   └── normalizer.py        # Key normalisation pipeline
└── integrations/
    ├── _pytest_plugin.py    # pytest fixture (auto-discovered)
    ├── _langsmith.py        # LangSmith evaluator adapter
    ├── _braintrust.py       # Braintrust scorer adapter
    └── _weave.py            # W&B Weave scorer adapter
```

## License

MIT
