"""
Tests for artifact persistence and re-analysis functionality.
"""

import pytest
import numpy as np
from neckenml.analyzer.reanalysis import (
    compute_derived_features,
    _recompute_folk_features,
    _recompute_swing_ratio,
    _recompute_feel
)


def test_recompute_folk_features():
    """Test folk feature recomputation from stored artifacts."""
    beat_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    beat_activations = [0.8, 0.6, 0.7, 0.9, 0.5, 0.6, 0.8, 0.7, 0.6, 0.9, 0.7, 0.8]
    intervals = np.diff(beat_times).tolist()

    features = _recompute_folk_features(beat_times, beat_activations, intervals)

    # Verify all expected keys exist
    assert "bpm" in features
    assert "avg_ibi" in features
    assert "punchiness" in features
    assert "r1_mean" in features
    assert "r2_mean" in features
    assert "r3_mean" in features
    assert "polska_score" in features
    assert "hambo_score" in features
    assert "bpm_stability" in features

    # Verify reasonable values
    assert 60 < features["bpm"] < 180  # Reasonable BPM range
    assert 0 <= features["punchiness"] <= 1
    assert 0 <= features["polska_score"] <= 1
    assert 0 <= features["hambo_score"] <= 1
    assert 0 <= features["bpm_stability"] <= 1


def test_recompute_swing_ratio():
    """Test swing ratio recomputation from stored onsets."""
    beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    onset_times = [0.0, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0]

    swing_ratio = _recompute_swing_ratio(beat_times, onset_times)

    # Verify reasonable swing ratio
    assert 0.5 <= swing_ratio <= 2.0


def test_recompute_swing_ratio_empty():
    """Test swing ratio with no onsets returns default."""
    beat_times = np.array([0.0, 0.5, 1.0])
    onset_times = []

    swing_ratio = _recompute_swing_ratio(beat_times, onset_times)
    assert swing_ratio == 1.0  # Default


def test_recompute_feel():
    """Test feel (articulation/bounciness) recomputation."""
    # Simulate a downsampled envelope
    envelope = np.random.rand(100) * 0.5 + 0.3  # Random envelope
    swing_ratio = 1.2

    articulation, bounciness = _recompute_feel(envelope.tolist(), swing_ratio)

    # Verify reasonable values
    assert 0 <= articulation <= 1
    assert 0 <= bounciness <= 1


def test_recompute_feel_empty():
    """Test feel with empty envelope returns zeros."""
    articulation, bounciness = _recompute_feel([], 1.0)
    assert articulation == 0.0
    assert bounciness == 0.0


def test_compute_derived_features_complete():
    """Test full feature derivation from complete artifact set."""
    # Create a complete artifact structure
    raw_artifacts = {
        "version": "1.0.0",
        "madmom": {
            "beat_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            "beat_info": [[t, i % 3 + 1] for i, t in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])],
            "ternary_confidence": 0.85,
            "bars": [0.5, 2.0, 3.5, 5.0]
        },
        "musicnn": {
            "avg_embedding": np.random.rand(200).tolist()  # 200-dim vector
        },
        "vocal": {
            "instrumental_score": 0.8,
            "vocal_score": 0.2
        },
        "audio_stats": {
            "loudness_lufs": -14.0,
            "rms": 0.15,
            "zcr": 0.05,
            "onset_rate": 2.5,
            "duration_seconds": 180.0
        },
        "onsets": {
            "librosa_onset_times": [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1, 5.6]
        },
        "structure": {
            "sections": [0.0, 16.0, 32.0],
            "section_labels": ["A", "B", "A"]
        },
        "dynamics": {
            "beat_activations": [0.8, 0.6, 0.7, 0.9, 0.5, 0.6, 0.8, 0.7, 0.6, 0.9, 0.7, 0.8],
            "envelope": np.random.rand(100).tolist(),
            "intervals": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        }
    }

    # Compute derived features
    features = compute_derived_features(raw_artifacts)

    # Verify all expected fields exist
    expected_fields = [
        "ml_suggested_style",
        "ml_confidence",
        "embedding",
        "loudness_lufs",
        "tempo_bpm",
        "bpm_stability",
        "is_likely_instrumental",
        "voice_probability",
        "swing_ratio",
        "articulation",
        "bounciness",
        "avg_beat_ratios",
        "punchiness",
        "polska_score",
        "hambo_score",
        "ternary_confidence",
        "meter",
        "bars",
        "beat_times",
        "sections",
        "section_labels",
        "folk_authenticity_score",
        "requires_manual_review",
        "folk_authenticity_breakdown",
        "folk_authenticity_interpretation"
    ]

    for field in expected_fields:
        assert field in features, f"Missing field: {field}"

    # Verify embedding has correct dimensionality
    assert len(features["embedding"]) == 217  # 200 + 9 + 1 + 3 + 1 + 1 + 1 + 1

    # Verify reasonable values
    assert features["is_likely_instrumental"] is True  # 0.8 > 0.2
    assert 0 <= features["ml_confidence"] <= 1
    assert 0 <= features["swing_ratio"] <= 2.0
    assert 0 <= features["articulation"] <= 1
    assert 0 <= features["bounciness"] <= 1


def test_compute_derived_features_minimal():
    """Test feature derivation with minimal artifacts."""
    raw_artifacts = {
        "version": "1.0.0",
        "madmom": {
            "beat_times": [],
            "beat_info": [],
            "ternary_confidence": 0.5,
            "bars": []
        },
        "musicnn": {
            "avg_embedding": np.zeros(200).tolist()
        },
        "vocal": {
            "instrumental_score": 0.5,
            "vocal_score": 0.5
        },
        "audio_stats": {
            "loudness_lufs": -14.0,
            "rms": 0.0,
            "zcr": 0.0,
            "onset_rate": 0.0,
            "duration_seconds": 0.0
        },
        "onsets": {},
        "structure": {},
        "dynamics": {}
    }

    # Should not crash with minimal data
    features = compute_derived_features(raw_artifacts)

    assert "ml_suggested_style" in features
    assert "embedding" in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
