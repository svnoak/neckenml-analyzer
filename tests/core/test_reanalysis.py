"""
Tests for the reanalysis module (compute_derived_features).

Tests cover:
- Basic functionality
- Output structure
- Edge cases with missing/empty data
- Feature computation accuracy
"""
import pytest
import numpy as np
from neckenml.core import compute_derived_features


class TestComputeDerivedFeatures:
    """Test suite for compute_derived_features function."""

    def test_returns_dict(self, sample_raw_artifacts):
        """Test that function returns a dictionary."""
        result = compute_derived_features(sample_raw_artifacts)
        assert isinstance(result, dict)

    def test_output_has_required_keys(self, sample_raw_artifacts):
        """Test that output contains all required keys."""
        result = compute_derived_features(sample_raw_artifacts)

        required_keys = [
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

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_embedding_dimensions(self, sample_raw_artifacts):
        """Test that embedding has correct dimensions (217)."""
        result = compute_derived_features(sample_raw_artifacts)

        # 200 (musicnn) + 9 (folk) + 1 (swing) + 3 (layout) + 1 (ternary)
        # + 1 (vocal) + 1 (articulation) + 1 (bounciness) = 217
        assert len(result['embedding']) == 217

    def test_tempo_calculation(self, sample_raw_artifacts):
        """Test that tempo is calculated correctly."""
        result = compute_derived_features(sample_raw_artifacts)

        # Tempo should be positive and in reasonable range
        assert result['tempo_bpm'] > 0
        assert 40 < result['tempo_bpm'] < 300

    def test_score_ranges(self, sample_raw_artifacts):
        """Test that scores are in valid ranges."""
        result = compute_derived_features(sample_raw_artifacts)

        # Scores should be 0-1
        assert 0.0 <= result['polska_score'] <= 1.0
        assert 0.0 <= result['hambo_score'] <= 1.0
        assert 0.0 <= result['folk_authenticity_score'] <= 1.0
        assert 0.0 <= result['bpm_stability'] <= 1.0

    def test_ml_confidence_range(self, sample_raw_artifacts):
        """Test that ML confidence is in valid range."""
        result = compute_derived_features(sample_raw_artifacts)

        assert 0.0 <= result['ml_confidence'] <= 1.0

    def test_vocal_detection(self, sample_raw_artifacts):
        """Test vocal detection based on scores."""
        result = compute_derived_features(sample_raw_artifacts)

        # Sample has vocal_score=0.2, instrumental_score=0.8
        assert result['is_likely_instrumental'] is True
        assert result['voice_probability'] == 0.2

    def test_avg_beat_ratios(self, sample_raw_artifacts):
        """Test that beat ratios sum to approximately 1."""
        result = compute_derived_features(sample_raw_artifacts)

        ratios = result['avg_beat_ratios']
        assert len(ratios) == 3
        assert abs(sum(ratios) - 1.0) < 0.1  # Should sum to ~1

    def test_empty_artifacts(self):
        """Test handling of empty artifacts."""
        empty_artifacts = {}

        result = compute_derived_features(empty_artifacts)

        # Should return result with default/zero values
        assert result is not None
        assert result['tempo_bpm'] == 0.0

    def test_missing_musicnn(self):
        """Test handling of missing MusiCNN embedding."""
        artifacts = {
            "rhythm_extractor": {
                "beats": [0.0, 0.5, 1.0, 1.5, 2.0],
                "bars": [],
                "ternary_confidence": 0.5
            },
            "vocal": {"vocal_score": 0.3, "instrumental_score": 0.7},
            "audio_stats": {"loudness_lufs": -14.0, "rms": 0.1, "zcr": 0.1, "onset_rate": 2.0},
            "onsets": {"librosa_onset_times": []},
            "dynamics": {"envelope": [], "beat_activations": [], "intervals": []}
        }

        result = compute_derived_features(artifacts)

        # Should still work with empty embedding
        assert result is not None

    def test_few_beats(self):
        """Test handling of very few beats."""
        artifacts = {
            "rhythm_extractor": {
                "beats": [0.0, 0.5, 1.0],  # Only 3 beats
                "bars": [],
                "ternary_confidence": 0.5
            },
            "musicnn": {"avg_embedding": list(np.zeros(200))},
            "vocal": {"vocal_score": 0.3, "instrumental_score": 0.7},
            "audio_stats": {"loudness_lufs": -14.0, "rms": 0.1, "zcr": 0.1, "onset_rate": 2.0},
            "onsets": {"librosa_onset_times": []},
            "dynamics": {"envelope": [], "beat_activations": [], "intervals": []}
        }

        result = compute_derived_features(artifacts)

        # Should return defaults for folk features
        assert result['polska_score'] == 0.0
        assert result['hambo_score'] == 0.0

    def test_structure_passthrough(self, sample_raw_artifacts):
        """Test that structure data is passed through correctly."""
        result = compute_derived_features(sample_raw_artifacts)

        assert result['sections'] == [[0, 8], [8, 16], [16, 24]]
        assert result['section_labels'] == ["A", "B", "A"]

    def test_bars_are_floats(self, sample_raw_artifacts):
        """Test that bars are converted to floats."""
        result = compute_derived_features(sample_raw_artifacts)

        for bar in result['bars']:
            assert isinstance(bar, float)

    def test_beat_times_are_floats(self, sample_raw_artifacts):
        """Test that beat times are converted to floats."""
        result = compute_derived_features(sample_raw_artifacts)

        for beat in result['beat_times']:
            assert isinstance(beat, float)

    def test_deterministic_output(self, sample_raw_artifacts):
        """Test that same input produces same output."""
        result1 = compute_derived_features(sample_raw_artifacts)
        result2 = compute_derived_features(sample_raw_artifacts)

        # Core numerical values should be identical
        assert result1['tempo_bpm'] == result2['tempo_bpm']
        assert result1['swing_ratio'] == result2['swing_ratio']
        assert result1['polska_score'] == result2['polska_score']
        assert result1['embedding'] == result2['embedding']


class TestPrivateHelpers:
    """Test private helper functions used by compute_derived_features."""

    def test_recompute_folk_features_sufficient_beats(self):
        """Test folk feature computation with sufficient beats."""
        from neckenml.core.reanalysis import _recompute_folk_features

        np.random.seed(42)
        beat_times = np.cumsum(np.random.normal(0.5, 0.02, 30))
        beat_activations = np.random.rand(30).tolist()
        intervals = np.diff(beat_times).tolist()

        result = _recompute_folk_features(beat_times, beat_activations, intervals)

        assert 'bpm' in result
        assert 'polska_score' in result
        assert 'hambo_score' in result
        assert result['bpm'] > 0

    def test_recompute_folk_features_insufficient_beats(self):
        """Test folk feature computation with insufficient beats."""
        from neckenml.core.reanalysis import _recompute_folk_features

        result = _recompute_folk_features(
            beat_times=np.array([0.0, 0.5, 1.0]),
            beat_activations=[],
            intervals=[]
        )

        # Should return defaults
        assert result['bpm'] == 0.0
        assert result['polska_score'] == 0.0

    def test_recompute_swing_ratio_no_onsets(self):
        """Test swing ratio with no onsets."""
        from neckenml.core.reanalysis import _recompute_swing_ratio

        result = _recompute_swing_ratio(
            beat_times=np.array([0.0, 0.5, 1.0, 1.5]),
            onset_times=[]
        )

        assert result == 1.0  # Default

    def test_recompute_feel_empty_envelope(self):
        """Test feel computation with empty envelope."""
        from neckenml.core.reanalysis import _recompute_feel

        articulation, bounciness = _recompute_feel(envelope=[], swing_ratio=1.0)

        assert articulation == 0.0
        assert bounciness == 0.0
