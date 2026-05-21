# Security Policy

## Supported versions

`json-semantic-diff` is currently **alpha** (`Development Status :: 3 - Alpha` in `pyproject.toml`). Only the latest `0.x` release receives security fixes. There is no LTS line yet; once a stable `1.0` ships this policy will be updated.

| Version | Supported |
| ------- | --------- |
| `0.x` (latest) | yes |
| anything older | no |

## Reporting a vulnerability

Please report vulnerabilities privately via [GitHub Security Advisories](https://github.com/mokhld/json-semantic-diff/security/advisories/new) ("Report a vulnerability" on the repo's Security tab). Do **not** open a public issue for security bugs.

When reporting, please include:

- A short description of the issue and its impact.
- Reproduction steps (a minimal JSON input + invocation is ideal).
- The version (`pip show json-semantic-diff`) and Python version.
- Any suggested fix or mitigation, if you have one.

You should expect an acknowledgement within a few working days. Fixes for confirmed issues land on `main` and are released in the next `0.x` patch.

## Scope

In scope:

- JSON parsing and tree construction in `tree/`.
- Embedding backends in `backends/` (static Levenshtein, FastEmbed local ONNX, OpenAI client).
- Optional cloud calls made by the OpenAI backend and by the LangSmith / Braintrust / Weave integrations under `integrations/`.
- The `json-semantic-diff` CLI entry point.

## Out of scope

- Resource exhaustion from intentionally pathological inputs (giant arrays, exponential string-length keys). Tree construction and the STED walk are both iterative (audit finding H1 fixed in wave 6) so deeply-nested JSON no longer hits `RecursionError`; however, total node count still drives O(n²) Hungarian cost-matrix work per nested OBJECT, and a 1 GiB JSON blob will still exhaust memory before being compared. Pre-validate size if you accept untrusted input.
- Vulnerabilities in third-party dependencies (numpy, scipy, fastembed, openai, etc.) — report those upstream. We will bump pins promptly once upstream fixes ship.

## Trust requirements for optional components

### `PersistentEmbeddingCache` (`pip install json-semantic-diff[diskcache]`)

The persistent cache uses `diskcache`, which stores values via Python `pickle`. Reading a cache directory whose contents were written by an untrusted party is equivalent to running `pickle.loads` on attacker-controlled bytes — **arbitrary code execution** in the process that calls `cache.embed()` or `cache.similarity()`.

Required practice:

- `cache_dir` MUST be a directory only writable by trusted principals on your machine or shared infrastructure.
- Do **not** share a `cache_dir` across mutually-untrusting tenants on NFS, S3-FUSE, multi-user CI volumes, or any filesystem where another principal could plant a malicious shard file.
- Per-user, per-host, or per-project cache directories are the safe default. The README example uses `~/.cache/json-semantic-diff/embeddings` for this reason.

### Eval-platform integrations (LangSmith, Braintrust, Weave)

The `integrations/` adapters upload `ComparisonResult` audit-trail fields — `key_mappings`, `unmatched_left`, `unmatched_right`, and `computation_time_ms` — to the third-party platform each time a score is recorded. JSON Pointer paths in those fields contain **raw key names from your data** (e.g. `/users/0/email`, `/payment/card_number`). If your JSON keys themselves are sensitive, wrap or redact the adapter output before evaluation.
