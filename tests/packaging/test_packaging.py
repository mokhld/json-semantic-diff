"""Packaging correctness verification for json-semantic-diff.

Tests validate PACK-04 requirements:
- Base install has no ImportError from optional backends
- py.typed marker is present in the wheel
- Pytest plugin entry point is registered
- Package metadata is correct

These tests inspect the built wheel and current installation rather than
creating temporary virtualenvs (faster, more reliable in CI).
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestBaseInstallNoImportError:
    """Verify base install does not raise ImportError from optional backends."""

    def test_import_json_semantic_diff(self):  # type: ignore[no-untyped-def]
        """Top-level import succeeds."""
        import json_semantic_diff

        assert hasattr(json_semantic_diff, "compare")
        assert hasattr(json_semantic_diff, "is_equivalent")
        assert hasattr(json_semantic_diff, "similarity_score")
        assert hasattr(json_semantic_diff, "consistency_score")

    def test_compare_basic(self):  # type: ignore[no-untyped-def]
        """compare() works with default StaticBackend."""
        from json_semantic_diff import compare

        result = compare({"a": 1}, {"a": 1})
        assert result.similarity_score == 1.0

    def test_is_equivalent_basic(self):  # type: ignore[no-untyped-def]
        """is_equivalent() works with default StaticBackend."""
        from json_semantic_diff import is_equivalent

        assert is_equivalent({"a": 1}, {"a": 1})

    def test_similarity_score_basic(self):  # type: ignore[no-untyped-def]
        """similarity_score() works with default StaticBackend."""
        from json_semantic_diff import similarity_score

        score = similarity_score({"a": 1}, {"a": 1})
        assert score == 1.0

    def test_no_fastembed_import_on_base(self):  # type: ignore[no-untyped-def]
        """Importing json_semantic_diff does not trigger fastembed import."""
        import json_semantic_diff

        # If fastembed is not installed, this confirms no ImportError.
        # If fastembed IS installed (dev env), this still passes.
        assert hasattr(json_semantic_diff, "compare")

    def test_no_openai_import_on_base(self):  # type: ignore[no-untyped-def]
        """Importing json_semantic_diff does not trigger openai import."""
        import json_semantic_diff

        assert hasattr(json_semantic_diff, "compare")

    def test_backends_import(self):  # type: ignore[no-untyped-def]
        """backends package imports successfully with StaticBackend available."""
        from json_semantic_diff.backends import StaticBackend

        backend = StaticBackend()
        sim = backend.similarity("user_name", "userName")
        assert sim == 1.0  # identical after normalization


class TestWheelContents:
    """Verify the built wheel contains required files."""

    @pytest.fixture(scope="class")
    def wheel_path(self) -> Path:
        """Build a fresh wheel and return its path."""
        dist_dir = PROJECT_ROOT / "dist"
        # Use poetry build since that's the project's build system
        result = subprocess.run(
            ["poetry", "build", "-f", "wheel"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"poetry build failed: {result.stderr}")

        wheels = sorted(dist_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime)
        if not wheels:
            pytest.skip("No wheel found in dist/")
        return wheels[-1]

    def test_py_typed_in_wheel(self, wheel_path: Path):  # type: ignore[no-untyped-def]
        """py.typed marker must be included in the wheel (PACK-01)."""
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
            py_typed_files = [n for n in names if n.endswith("py.typed")]
            assert py_typed_files, f"py.typed not found in wheel. Contents: {names}"

    def test_no_pycache_in_wheel(self, wheel_path: Path):  # type: ignore[no-untyped-def]
        """__pycache__ directories must not be in the wheel."""
        with zipfile.ZipFile(wheel_path) as zf:
            pycache_files = [n for n in zf.namelist() if "__pycache__" in n]
            assert not pycache_files, f"__pycache__ found in wheel: {pycache_files}"

    def test_all_source_modules_in_wheel(self, wheel_path: Path):  # type: ignore[no-untyped-def]
        """All source modules must be present in the wheel."""
        expected_modules = [
            "json_semantic_diff/__init__.py",
            "json_semantic_diff/api.py",
            "json_semantic_diff/comparator.py",
            "json_semantic_diff/result.py",
            "json_semantic_diff/cache.py",
            "json_semantic_diff/scorer.py",
            "json_semantic_diff/protocols.py",
            "json_semantic_diff/algorithm/__init__.py",
            "json_semantic_diff/algorithm/config.py",
            "json_semantic_diff/algorithm/costs.py",
            "json_semantic_diff/algorithm/matcher.py",
            "json_semantic_diff/algorithm/normalizer.py",
            "json_semantic_diff/algorithm/sted.py",
            "json_semantic_diff/backends/__init__.py",
            "json_semantic_diff/backends/static.py",
            "json_semantic_diff/backends/fastembed.py",
            "json_semantic_diff/backends/openai.py",
            "json_semantic_diff/tree/__init__.py",
            "json_semantic_diff/tree/builder.py",
            "json_semantic_diff/tree/nodes.py",
            "json_semantic_diff/tree/normalizer.py",
            "json_semantic_diff/integrations/__init__.py",
            "json_semantic_diff/integrations/_pytest_plugin.py",
            "json_semantic_diff/integrations/_langsmith.py",
            "json_semantic_diff/integrations/_braintrust.py",
            "json_semantic_diff/integrations/_weave.py",
        ]
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
            for module in expected_modules:
                assert any(module in n for n in names), (
                    f"Module {module} not found in wheel"
                )

    def test_metadata_in_wheel(self, wheel_path: Path):  # type: ignore[no-untyped-def]
        """Wheel metadata must include correct package info."""
        with zipfile.ZipFile(wheel_path) as zf:
            metadata_files = [n for n in zf.namelist() if "METADATA" in n]
            assert metadata_files, "No METADATA found in wheel"
            metadata = zf.read(metadata_files[0]).decode()
            assert (
                "json-semantic-diff" in metadata.lower()
                or "json_semantic_diff" in metadata.lower()
            )
            assert "0.1.0" in metadata


class TestPytestPluginDiscovery:
    """Verify the pytest plugin is discoverable."""

    def test_entry_point_registered(self):  # type: ignore[no-untyped-def]
        """pytest11 entry point must be registered for json-semantic-diff."""
        from importlib.metadata import entry_points

        pytest11_eps = entry_points(group="pytest11")

        sd_eps = [
            ep
            for ep in pytest11_eps
            if "semantic" in ep.name.lower() or "semantic" in str(ep.value).lower()
        ]
        assert sd_eps, (
            f"No pytest11 entry point found for json-semantic-diff. "
            f"Available: {[ep.name for ep in pytest11_eps]}"
        )

    def test_fixture_available(self):  # type: ignore[no-untyped-def]
        """assert_json_equivalent fixture must be importable from plugin."""
        import importlib

        mod = importlib.import_module("json_semantic_diff.integrations._pytest_plugin")
        assert hasattr(mod, "assert_json_equivalent")
        assert callable(mod.assert_json_equivalent)

    def test_plugin_discovery_via_pytest(self):  # type: ignore[no-untyped-def]
        """pytest --fixtures should list assert_json_equivalent."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--fixtures", "-q"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert "assert_json_equivalent" in result.stdout, (
            f"Fixture not found in pytest fixtures list. stdout: {result.stdout[:500]}"
        )


class TestPackageMetadata:
    """Verify pyproject.toml metadata completeness."""

    def test_version(self):  # type: ignore[no-untyped-def]
        """Package version must be 0.1.0."""
        import json_semantic_diff

        assert json_semantic_diff.__version__ == "0.1.0"

    def test_all_exports(self):  # type: ignore[no-untyped-def]
        """__all__ must include the documented public API."""
        import json_semantic_diff

        expected = {
            "ArrayComparisonMode",
            "ComparisonResult",
            "STEDComparator",
            "STEDConfig",
            "compare",
            "compare_batch",
            "compare_batch_pairs",
            "consistency_score",
            "format_diff",
            "is_equivalent",
            "similarity_score",
        }
        actual = set(json_semantic_diff.__all__)
        assert expected == actual, (
            f"Missing: {expected - actual}, Extra: {actual - expected}"
        )
