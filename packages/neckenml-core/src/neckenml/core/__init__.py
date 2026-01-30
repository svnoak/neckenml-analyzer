"""
neckenml Core - MIT licensed components for Swedish folk music classification.

This module contains the database models, classifier logic, and training utilities
that do NOT depend on AGPL-licensed audio processing libraries (like Essentia).

Users can use this module freely under the MIT license. If you need audio analysis,
install the full package:
    pip install neckenml-analyzer

Usage:
    from neckenml.core import StyleClassifier, Track, TrainingService
    from neckenml.core import compute_derived_features  # Re-analyze from stored artifacts
"""

__version__ = "0.3.0"

from neckenml.core.models import Track, AnalysisSource, TrackDanceStyle, Base
from neckenml.core.classifier import StyleClassifier, ClassificationHead, ClassifierParams
from neckenml.core.sources import AudioSource, FileAudioSource
from neckenml.core.training import TrainingService
from neckenml.core.folk_authenticity import FolkAuthenticityDetector
from neckenml.core.reanalysis import compute_derived_features

__all__ = [
    # Database models
    "Track",
    "AnalysisSource",
    "TrackDanceStyle",
    "Base",
    # Classifier
    "StyleClassifier",
    "ClassificationHead",
    "ClassifierParams",
    # Sources
    "AudioSource",
    "FileAudioSource",
    # Training
    "TrainingService",
    # Re-analysis (from stored artifacts, no audio needed)
    "FolkAuthenticityDetector",
    "compute_derived_features",
]
