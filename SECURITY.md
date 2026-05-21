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

- **Malicious JSON inputs causing recursion errors / stack overflow.** The current tree builder is recursive, so deeply nested JSON (roughly past Python's default recursion limit, ~1000 levels) will raise `RecursionError`. This is tracked as audit finding H1 and is a known issue. Until the planned iterative-walk fix lands, callers MUST validate input depth (and ideally total node count) before passing untrusted JSON to `compare()` / `similarity_score()`.
- Resource exhaustion from intentionally pathological inputs (giant arrays, exponential string-length keys). Pre-validate size if you accept untrusted input.
- Vulnerabilities in third-party dependencies (numpy, scipy, fastembed, openai, etc.) — report those upstream. We will bump pins promptly once upstream fixes ship.
