"""
Test that neckenml-analyzer imports work correctly.

Note: These tests require the AGPL dependencies (essentia, etc.) to be installed.
They may be skipped in environments without audio processing dependencies.
"""
import pytest


class TestAnalyzerImports:
    """Test that analyzer package imports work."""

    def test_import_analyzer_package(self):
        """Test importing the analyzer package."""
        try:
            from neckenml import analyzer
            assert analyzer is not None
        except ImportError as e:
            pytest.skip(f"Analyzer dependencies not installed: {e}")

    def test_import_audio_analyzer(self):
        """Test importing AudioAnalyzer."""
        try:
            from neckenml.analyzer import AudioAnalyzer
            assert AudioAnalyzer is not None
        except ImportError as e:
            pytest.skip(f"Audio analyzer dependencies not installed: {e}")

    def test_all_exports_listed(self):
        """Test that __all__ contains expected exports."""
        try:
            from neckenml.analyzer import __all__
            assert "AudioAnalyzer" in __all__
        except ImportError as e:
            pytest.skip(f"Analyzer dependencies not installed: {e}")

    def test_folk_authenticity_moved_to_core(self):
        """Test that FolkAuthenticityDetector is NOT in analyzer anymore."""
        try:
            from neckenml.analyzer import __all__
            assert "FolkAuthenticityDetector" not in __all__
        except ImportError as e:
            pytest.skip(f"Analyzer dependencies not installed: {e}")

    def test_compute_derived_features_moved_to_core(self):
        """Test that compute_derived_features is NOT in analyzer anymore."""
        try:
            from neckenml.analyzer import __all__
            assert "compute_derived_features" not in __all__
        except ImportError as e:
            pytest.skip(f"Analyzer dependencies not installed: {e}")
