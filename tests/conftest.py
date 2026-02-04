"""
Shared pytest fixtures for neckenml tests.
"""
import pytest
import numpy as np


@pytest.fixture
def sample_embedding():
    """A sample 200-dimensional MusiCNN-style embedding."""
    np.random.seed(42)
    return np.random.randn(200).astype(np.float32)


@pytest.fixture
def sample_raw_artifacts():
    """Sample raw artifacts as stored by the audio analyzer."""
    np.random.seed(42)

    # Generate realistic beat times (120 BPM = 0.5s per beat)
    beat_times = np.cumsum(np.random.normal(0.5, 0.02, 60))

    return {
        "rhythm_extractor": {
            "beats": beat_times.tolist(),
            "bars": [beat_times[i] for i in range(0, len(beat_times), 3)],
            "ternary_confidence": 0.85,
            "beat_info": [[t, (i % 3) + 1] for i, t in enumerate(beat_times)]
        },
        "musicnn": {
            "avg_embedding": np.random.randn(200).tolist()
        },
        "vocal": {
            "vocal_score": 0.2,
            "instrumental_score": 0.8
        },
        "audio_stats": {
            "loudness_lufs": -14.0,
            "rms": 0.15,
            "zcr": 0.08,
            "onset_rate": 2.5
        },
        "onsets": {
            "librosa_onset_times": (beat_times + np.random.normal(0, 0.05, len(beat_times))).tolist()
        },
        "dynamics": {
            "envelope": np.abs(np.random.randn(1000)).tolist(),
            "beat_activations": np.random.rand(60).tolist(),
            "intervals": np.diff(beat_times).tolist()
        },
        "structure": {
            "sections": [[0, 8], [8, 16], [16, 24]],
            "section_labels": ["A", "B", "A"]
        }
    }


@pytest.fixture
def sample_analysis_features():
    """Sample analysis features as returned by compute_derived_features."""
    return {
        "ml_suggested_style": "Polska",
        "ml_confidence": 0.85,
        "embedding": list(np.random.randn(217)),
        "loudness_lufs": -14.0,
        "tempo_bpm": 120.0,
        "bpm_stability": 0.92,
        "is_likely_instrumental": True,
        "voice_probability": 0.2,
        "swing_ratio": 1.15,
        "articulation": 0.45,
        "bounciness": 0.55,
        "avg_beat_ratios": [0.33, 0.34, 0.33],
        "punchiness": 0.4,
        "polska_score": 0.35,
        "hambo_score": 0.25,
        "ternary_confidence": 0.85,
        "meter": "3/4",
        "bars": [0.0, 1.5, 3.0, 4.5],
        "beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "sections": [[0, 8], [8, 16]],
        "section_labels": ["A", "B"],
        "folk_authenticity_score": 0.75,
        "requires_manual_review": False,
        "folk_authenticity_breakdown": {
            "dynamics": 0.8,
            "timbre": 0.7,
            "timing": 0.6,
            "articulation": 0.8,
            "vocals": 0.5,
            "production": 0.7
        },
        "folk_authenticity_interpretation": "Likely traditional folk"
    }


@pytest.fixture
def mock_track():
    """A mock Track object for testing classification."""
    class MockTrack:
        def __init__(self):
            self.id = "test-track-id"
            self.title = "Slängpolska från Boda"
            self.artist = "Boda Spelmanslag"
            self.dance_styles = []
            self.artist_links = []
            self.album_links = []

    return MockTrack()


@pytest.fixture
def mock_db_session():
    """A mock database session for testing."""
    class MockSession:
        def __init__(self):
            self.added = []
            self.committed = False
            self.rolled_back = False

        def add(self, obj):
            self.added.append(obj)

        def commit(self):
            self.committed = True

        def rollback(self):
            self.rolled_back = True

        def query(self, model):
            return MockQuery()

    class MockQuery:
        def filter(self, *args, **kwargs):
            return self

        def filter_by(self, **kwargs):
            return self

        def first(self):
            return None

        def all(self):
            return []

    return MockSession()
