"""
Test that all neckenml-core imports work correctly.

These tests verify the package structure is correct and all public APIs
are accessible.
"""
import pytest


class TestCoreImports:
    """Test that core package imports work."""

    def test_import_core_package(self):
        """Test importing the main core package."""
        from neckenml import core
        assert core is not None
        assert hasattr(core, '__version__')

    def test_import_database_models(self):
        """Test importing database models."""
        from neckenml.core import Track, AnalysisSource, TrackDanceStyle, Base

        assert Track is not None
        assert AnalysisSource is not None
        assert TrackDanceStyle is not None
        assert Base is not None

    def test_import_classifier(self):
        """Test importing classifier components."""
        from neckenml.core import StyleClassifier, ClassificationHead, ClassifierParams

        assert StyleClassifier is not None
        assert ClassificationHead is not None
        assert ClassifierParams is not None

    def test_import_sources(self):
        """Test importing audio source components."""
        from neckenml.core import AudioSource, FileAudioSource

        assert AudioSource is not None
        assert FileAudioSource is not None

    def test_import_training(self):
        """Test importing training components."""
        from neckenml.core import TrainingService

        assert TrainingService is not None

    def test_import_folk_authenticity(self):
        """Test importing folk authenticity detector."""
        from neckenml.core import FolkAuthenticityDetector

        assert FolkAuthenticityDetector is not None

    def test_import_reanalysis(self):
        """Test importing reanalysis functions."""
        from neckenml.core import compute_derived_features

        assert compute_derived_features is not None
        assert callable(compute_derived_features)

    def test_all_exports_listed(self):
        """Test that __all__ contains expected exports."""
        from neckenml.core import __all__

        expected = [
            "Track",
            "AnalysisSource",
            "TrackDanceStyle",
            "Base",
            "StyleClassifier",
            "ClassificationHead",
            "ClassifierParams",
            "AudioSource",
            "FileAudioSource",
            "TrainingService",
            "FolkAuthenticityDetector",
            "compute_derived_features",
        ]

        for item in expected:
            assert item in __all__, f"{item} not in __all__"
