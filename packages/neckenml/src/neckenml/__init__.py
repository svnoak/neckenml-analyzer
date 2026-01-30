"""
neckenml - Swedish folk music audio analysis and dance style classification.

This is a convenience package that installs both:
- neckenml-core (MIT) - Classification and database models
- neckenml-analyzer (AGPL) - Audio analysis with Essentia

License: AGPL-3.0 (due to neckenml-analyzer dependency)

For MIT-only usage, install neckenml-core directly instead:
    pip install neckenml-core

Usage:
    from neckenml import AudioAnalyzer, StyleClassifier

    # Analyze audio
    with AudioAnalyzer() as analyzer:
        result = analyzer.analyze_file("track.mp3")

    # Classify with pre-computed features
    classifier = StyleClassifier()
    styles = classifier.classify(track, result)
"""

__version__ = "0.3.0"

# Re-export from core (MIT)
from neckenml.core import (
    Track,
    AnalysisSource,
    TrackDanceStyle,
    Base,
    StyleClassifier,
    ClassificationHead,
    ClassifierParams,
    AudioSource,
    FileAudioSource,
    TrainingService,
    FolkAuthenticityDetector,
    compute_derived_features,
)

# Re-export from analyzer (AGPL)
from neckenml.analyzer import (
    AudioAnalyzer,
)

__all__ = [
    # Core (MIT)
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
    # Analyzer (AGPL)
    "AudioAnalyzer",
]
