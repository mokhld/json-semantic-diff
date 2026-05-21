"""Tests for the ``json_semantic_diff._cli`` console script entry point.

These tests call :func:`main` directly with an explicit argv list and assert on
stdout/stderr via pytest's ``capsys`` fixture.  No subprocesses are spawned —
the CLI is fully library-callable.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from json_semantic_diff._cli import main


def _write(tmp_path: Path, name: str, payload: object) -> str:
    """Write ``payload`` as JSON under ``tmp_path/name`` and return the path."""
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestBasic:
    """Smoke tests for the default (no-flag) CLI invocation."""

    def test_identical_docs_print_float_in_range(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1, "b": 2})
        right = _write(tmp_path, "right.json", {"a": 1, "b": 2})

        rc = main([left, right])

        assert rc == 0
        captured = capsys.readouterr()
        score = float(captured.out.strip())
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(1.0)
        assert captured.err == ""

    def test_different_docs_print_lower_score(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        right = _write(tmp_path, "right.json", {"completely": "different"})

        rc = main([left, right])

        assert rc == 0
        score = float(capsys.readouterr().out.strip())
        assert 0.0 <= score < 1.0


class TestJsonFlag:
    """Tests for the ``--json`` audit-trail output mode."""

    def test_json_flag_prints_valid_json_with_expected_keys(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"name": "Alice", "age": 30})
        right = _write(tmp_path, "right.json", {"name": "Alice", "age": 31})

        rc = main(["--json", left, right])

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "similarity_score" in payload
        assert "key_mappings" in payload
        assert "matched_pairs" in payload
        assert "unmatched_left" in payload
        assert "unmatched_right" in payload
        assert "computation_time_ms" in payload
        assert 0.0 <= payload["similarity_score"] <= 1.0


class TestThreshold:
    """Tests for the ``--threshold`` exit-code mode."""

    def test_identical_docs_meet_threshold(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        right = _write(tmp_path, "right.json", {"a": 1})

        rc = main(["--threshold", "0.5", left, right])

        assert rc == 0
        # Threshold mode is silent by default.
        assert capsys.readouterr().out == ""

    def test_different_docs_fail_full_threshold(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        right = _write(tmp_path, "right.json", {"b": 2})

        rc = main(["--threshold", "1.0", left, right])

        assert rc == 1
        assert capsys.readouterr().out == ""

    def test_threshold_verbose_prints_score(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        right = _write(tmp_path, "right.json", {"a": 1})

        rc = main(["--threshold", "0.5", "--verbose", left, right])

        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert out != ""
        score = float(out)
        assert 0.0 <= score <= 1.0


class TestErrorHandling:
    """Tests for the ``exit 2`` family of usage / input errors."""

    def test_missing_file_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        missing = str(tmp_path / "does-not-exist.json")

        rc = main([left, missing])

        assert rc == 2
        err = capsys.readouterr().err
        assert err.startswith("error: cannot read ")
        assert "does-not-exist.json" in err

    def test_malformed_json_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")

        rc = main([left, str(bad)])

        assert rc == 2
        err = capsys.readouterr().err
        assert err.startswith("error: invalid JSON in ")
        assert "bad.json" in err

    def test_invalid_weight_combo_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        right = _write(tmp_path, "right.json", {"a": 1})

        rc = main(
            [
                "--structural-weight",
                "0.7",
                "--content-weight",
                "0.7",
                left,
                right,
            ]
        )

        assert rc == 2
        err = capsys.readouterr().err
        assert err.startswith("error: ")
        # Either a w_s/w_c range error or a sum error — both flag the misuse.
        assert "w_" in err

    def test_both_stdin_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["-", "-"])

        assert rc == 2
        err = capsys.readouterr().err
        assert "stdin" in err.lower()


class TestVersion:
    """Tests for the ``--version`` flag."""

    def test_version_flag_prints_and_exits_0(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # argparse's ``action="version"`` raises SystemExit(0).
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])

        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        # argparse writes the version banner to stdout.
        assert "json-semantic-diff" in captured.out


class TestStdin:
    """Tests for reading one document from stdin via ``-``."""

    def test_stdin_left_works(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        right = _write(tmp_path, "right.json", {"a": 1})
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"a": 1})))

        rc = main(["-", right])

        assert rc == 0
        score = float(capsys.readouterr().out.strip())
        assert score == pytest.approx(1.0)

    def test_stdin_right_works(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"a": 1})))

        rc = main([left, "-"])

        assert rc == 0
        score = float(capsys.readouterr().out.strip())
        assert score == pytest.approx(1.0)


class TestWeights:
    """Tests for the structural / content weight knobs."""

    def test_custom_weights_summing_to_one_accepted(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        right = _write(tmp_path, "right.json", {"a": 1})

        rc = main(
            [
                "--structural-weight",
                "0.7",
                "--content-weight",
                "0.3",
                left,
                right,
            ]
        )

        assert rc == 0
        score = float(capsys.readouterr().out.strip())
        assert 0.0 <= score <= 1.0

    def test_weights_not_summing_to_one_exit_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        left = _write(tmp_path, "left.json", {"a": 1})
        right = _write(tmp_path, "right.json", {"a": 1})

        rc = main(
            [
                "--structural-weight",
                "0.4",
                "--content-weight",
                "0.4",
                left,
                right,
            ]
        )

        assert rc == 2
        err = capsys.readouterr().err
        assert err.startswith("error: ")


class TestArrayMode:
    """Tests for the ``--array-mode`` knob."""

    @pytest.mark.parametrize("mode", ["auto", "ordered", "unordered"])
    def test_array_mode_accepted(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        mode: str,
    ) -> None:
        left = _write(tmp_path, "left.json", [1, 2, 3])
        right = _write(tmp_path, "right.json", [3, 2, 1])

        rc = main(["--array-mode", mode, left, right])

        assert rc == 0
        score = float(capsys.readouterr().out.strip())
        assert 0.0 <= score <= 1.0
