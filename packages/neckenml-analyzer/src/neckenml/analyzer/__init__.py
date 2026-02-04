"""
Audio analysis components for Swedish folk music.

AGPL-3.0 licensed - uses Essentia for audio processing.

Note: FolkAuthenticityDetector and compute_derived_features are in neckenml-core
(MIT licensed) since they don't depend on Essentia.
"""

from neckenml.analyzer.audio_analyzer import AudioAnalyzer

__all__ = ["AudioAnalyzer"]
