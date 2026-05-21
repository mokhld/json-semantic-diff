"""Command-line interface for json-semantic-diff.

Provides the ``json-semantic-diff`` console script.  The orchestrator wires up
``[project.scripts]`` so that ``json-semantic-diff = "json_semantic_diff._cli:main"``.

The module exposes a single public entry point, :func:`main`, which accepts an
``argv`` list (for testability) and returns an exit code.  No ``sys.exit`` is
invoked from inside ``main`` itself — the ``__main__`` guard at the bottom
performs that.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from typing import Any, TextIO

from json_semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig
from json_semantic_diff.api import compare, is_equivalent

__all__ = ["main"]


_DESCRIPTION = """\
Score the semantic similarity between two JSON documents.

Examples:
  json-semantic-diff left.json right.json
  json-semantic-diff --json left.json right.json
  json-semantic-diff --threshold 0.95 left.json right.json
  cat left.json | json-semantic-diff - right.json
  json-semantic-diff --structural-weight 0.7 --content-weight 0.3 a.json b.json
"""


def _build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser used by :func:`main`.

    Kept as a separate function so tests can exercise help/version paths
    without going through ``main`` argv plumbing.
    """
    parser = argparse.ArgumentParser(
        prog="json-semantic-diff",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "left",
        help="Path to the left JSON document (use '-' to read from stdin).",
    )
    parser.add_argument(
        "right",
        help="Path to the right JSON document (use '-' to read from stdin).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print the full ComparisonResult as JSON (audit trail) instead of just the score.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Exit 0 when similarity >= THRESHOLD, else exit 1.  "
            "Suppresses score output unless --verbose is set."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In threshold mode, also print the similarity score on stdout.",
    )
    parser.add_argument(
        "--structural-weight",
        type=float,
        default=0.5,
        metavar="FLOAT",
        dest="structural_weight",
        help="Structural weight w_s in [0, 1] (default 0.5).  Must sum with --content-weight to 1.0.",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=0.5,
        metavar="FLOAT",
        dest="content_weight",
        help="Content weight w_c in [0, 1] (default 0.5).  Must sum with --structural-weight to 1.0.",
    )
    parser.add_argument(
        "--array-mode",
        choices=("auto", "ordered", "unordered"),
        default="auto",
        dest="array_mode",
        help="How JSON arrays are compared (default: auto).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"json-semantic-diff {_get_version()}",
        help="Print the installed package version and exit.",
    )
    return parser


def _get_version() -> str:
    """Return the installed package version, falling back to 'unknown'.

    Uses ``importlib.metadata`` so the wheel metadata is the single source of
    truth.  Falls back gracefully if the package is not yet installed (e.g.
    running directly from a source checkout without ``pip install -e .``).
    """
    try:
        return metadata.version("json-semantic-diff")
    except metadata.PackageNotFoundError:
        return "unknown"


def _read_json(
    path: str,
    *,
    stdin: TextIO,
    stderr: TextIO,
) -> tuple[Any, int]:
    """Read JSON from ``path`` (or stdin when ``path == "-"``).

    Returns a ``(value, exit_code)`` pair.  When ``exit_code`` is non-zero,
    ``value`` is ``None`` and an error has already been written to ``stderr``.
    """
    label = "<stdin>" if path == "-" else path
    try:
        if path == "-":
            text = stdin.read()
        else:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
    except OSError as exc:
        print(f"error: cannot read {label}: {exc.strerror or exc}", file=stderr)
        return None, 2

    try:
        return json.loads(text), 0
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {label}: {exc}", file=stderr)
        return None, 2


def _array_mode_from_str(value: str) -> ArrayComparisonMode:
    """Map the CLI ``--array-mode`` flag onto the enum value."""
    # StrEnum values are lowercase auto()-generated strings; the CLI choices
    # mirror those exactly so a direct construction works.
    return ArrayComparisonMode(value)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (excluding the program name).  When ``None``,
            ``sys.argv[1:]`` is used by argparse.

    Returns:
        Process exit code.  0 = success / equivalent; 1 = below threshold;
        2 = usage or input error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    stdout: TextIO = sys.stdout
    stderr: TextIO = sys.stderr
    stdin: TextIO = sys.stdin

    if args.left == "-" and args.right == "-":
        print(
            "error: cannot read both LEFT and RIGHT from stdin ('-' used twice)",
            file=stderr,
        )
        return 2

    try:
        config = STEDConfig(
            w_s=args.structural_weight,
            w_c=args.content_weight,
            array_comparison_mode=_array_mode_from_str(args.array_mode),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=stderr)
        return 2

    left, rc = _read_json(args.left, stdin=stdin, stderr=stderr)
    if rc != 0:
        return rc
    right, rc = _read_json(args.right, stdin=stdin, stderr=stderr)
    if rc != 0:
        return rc

    if args.threshold is not None:
        try:
            equivalent = is_equivalent(
                left, right, threshold=args.threshold, config=config
            )
        except ValueError as exc:
            print(f"error: {exc}", file=stderr)
            return 2
        if args.verbose:
            # Re-run compare to surface the actual score; cheap relative to
            # threshold-only path's value to humans debugging.
            result = compare(left, right, config=config)
            print(result.similarity_score, file=stdout)
        return 0 if equivalent else 1

    try:
        result = compare(left, right, config=config)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=stderr)
        return 2

    if args.json_output:
        print(result.to_json(indent=2), file=stdout)
    else:
        print(result.similarity_score, file=stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via console_scripts
    sys.exit(main())
