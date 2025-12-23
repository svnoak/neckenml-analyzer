"""
neckenml Analyzer - Swedish folk music audio analysis and dance style classification.
"""

from neckenml.analyzer.audio_analyzer import AudioAnalyzer
from neckenml.classifier.style_classifier import StyleClassifier
from neckenml.classifier.style_head import ClassificationHead
from neckenml.analyzer.folk_authenticity import FolkAuthenticityDetector
from neckenml.analyzer.reanalysis import compute_derived_features

__version__ = "0.2.0"

__all__ = [
    "AudioAnalyzer",
    "StyleClassifier",
    "ClassificationHead",
    "FolkAuthenticityDetector",
    "compute_derived_features",
]
