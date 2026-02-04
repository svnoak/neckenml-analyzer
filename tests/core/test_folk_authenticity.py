"""
Tests for the FolkAuthenticityDetector.

Tests cover:
- Basic functionality
- Edge cases
- Score ranges
- Interpretation strings
"""
import pytest
import numpy as np
from neckenml.core import FolkAuthenticityDetector


class TestFolkAuthenticityDetector:
    """Test suite for FolkAuthenticityDetector."""

    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return FolkAuthenticityDetector(manual_review_threshold=0.6)

    def test_initialization(self):
        """Test detector can be initialized."""
        detector = FolkAuthenticityDetector()
        assert detector.manual_review_threshold == 0.6

    def test_custom_threshold(self):
        """Test detector with custom threshold."""
        detector = FolkAuthenticityDetector(manual_review_threshold=0.5)
        assert detector.manual_review_threshold == 0.5

    def test_analyze_returns_dict(self, detector, sample_embedding):
        """Test that analyze returns expected dict structure."""
        result = detector.analyze(
            rms_value=0.5,
            zcr_value=0.1,
            swing_ratio=0.6,
            articulation="smooth",
            bounciness=0.5,
            voice_probability=0.3,
            is_likely_instrumental=True,
            embedding=sample_embedding
        )

        assert isinstance(result, dict)
        assert 'folk_authenticity_score' in result
        assert 'requires_manual_review' in result
        assert 'confidence_breakdown' in result
        assert 'interpretation' in result

    def test_score_range(self, detector, sample_embedding):
        """Test that score is always between 0 and 1."""
        # Test with various inputs
        test_cases = [
            {"rms_value": 0.1, "zcr_value": 0.05},
            {"rms_value": 0.9, "zcr_value": 0.3},
            {"rms_value": 0.5, "zcr_value": 0.1},
        ]

        for case in test_cases:
            result = detector.analyze(
                **case,
                swing_ratio=0.6,
                articulation="smooth",
                bounciness=0.5,
                voice_probability=0.3,
                is_likely_instrumental=True,
                embedding=sample_embedding
            )

            assert 0.0 <= result['folk_authenticity_score'] <= 1.0

    def test_traditional_folk_characteristics(self, detector, sample_embedding):
        """Test that traditional folk characteristics score higher."""
        # Traditional folk: moderate RMS, acoustic ZCR, natural swing
        traditional_result = detector.analyze(
            rms_value=0.5,
            zcr_value=0.1,
            swing_ratio=0.6,
            articulation="smooth",
            bounciness=0.5,
            voice_probability=0.3,
            is_likely_instrumental=True,
            embedding=sample_embedding
        )

        # Modern electronic: high RMS, extreme ZCR, quantized timing
        modern_result = detector.analyze(
            rms_value=0.9,
            zcr_value=0.4,
            swing_ratio=0.5,
            articulation="staccato",
            bounciness=0.9,
            voice_probability=0.1,
            is_likely_instrumental=True,
            embedding=sample_embedding
        )

        assert traditional_result['folk_authenticity_score'] > modern_result['folk_authenticity_score']

    def test_manual_review_flag(self, detector, sample_embedding):
        """Test that manual review flag is set correctly."""
        # High score - no review needed
        high_score_result = detector.analyze(
            rms_value=0.5,
            zcr_value=0.1,
            swing_ratio=0.6,
            articulation="smooth",
            bounciness=0.5,
            voice_probability=0.6,
            is_likely_instrumental=False,
            embedding=sample_embedding
        )

        # Low score - review needed
        low_score_result = detector.analyze(
            rms_value=0.95,
            zcr_value=0.5,
            swing_ratio=0.5,
            articulation="staccato",
            bounciness=0.95,
            voice_probability=0.1,
            is_likely_instrumental=True,
            embedding=sample_embedding
        )

        # The manual review flag should correlate with score threshold
        if high_score_result['folk_authenticity_score'] >= 0.6:
            assert high_score_result['requires_manual_review'] is False

        if low_score_result['folk_authenticity_score'] < 0.6:
            assert low_score_result['requires_manual_review'] is True

    def test_confidence_breakdown_keys(self, detector, sample_embedding):
        """Test that confidence breakdown has expected keys."""
        result = detector.analyze(
            rms_value=0.5,
            zcr_value=0.1,
            swing_ratio=0.6,
            articulation="smooth",
            bounciness=0.5,
            voice_probability=0.3,
            is_likely_instrumental=True,
            embedding=sample_embedding
        )

        breakdown = result['confidence_breakdown']
        expected_keys = ['dynamics', 'timbre', 'timing', 'articulation', 'vocals', 'production']

        for key in expected_keys:
            assert key in breakdown, f"Missing key: {key}"
            assert 0.0 <= breakdown[key] <= 1.0, f"Key {key} out of range"

    def test_interpretation_strings(self, detector, sample_embedding):
        """Test that interpretation returns valid strings."""
        valid_interpretations = [
            "Very likely traditional folk",
            "Likely traditional folk",
            "Uncertain - may be modern folk or folk-pop",
            "Likely modern/electronic production",
            "Very likely modern/electronic music"
        ]

        result = detector.analyze(
            rms_value=0.5,
            zcr_value=0.1,
            swing_ratio=0.6,
            articulation="smooth",
            bounciness=0.5,
            voice_probability=0.3,
            is_likely_instrumental=True,
            embedding=sample_embedding
        )

        assert result['interpretation'] in valid_interpretations

    def test_instrumental_vs_vocal_handling(self, detector, sample_embedding):
        """Test that instrumental vs vocal tracks are handled correctly."""
        # Instrumental track
        instrumental_result = detector.analyze(
            rms_value=0.5,
            zcr_value=0.1,
            swing_ratio=0.6,
            articulation="smooth",
            bounciness=0.5,
            voice_probability=0.1,
            is_likely_instrumental=True,
            embedding=sample_embedding
        )

        # Vocal track
        vocal_result = detector.analyze(
            rms_value=0.5,
            zcr_value=0.1,
            swing_ratio=0.6,
            articulation="smooth",
            bounciness=0.5,
            voice_probability=0.8,
            is_likely_instrumental=False,
            embedding=sample_embedding
        )

        # Instrumental should use neutral vocal score
        assert instrumental_result['confidence_breakdown']['vocals'] == 0.5

    def test_empty_embedding(self, detector):
        """Test handling of empty or short embedding."""
        result = detector.analyze(
            rms_value=0.5,
            zcr_value=0.1,
            swing_ratio=0.6,
            articulation="smooth",
            bounciness=0.5,
            voice_probability=0.3,
            is_likely_instrumental=True,
            embedding=[]
        )

        # Should fall back to neutral production score
        assert result['confidence_breakdown']['production'] == 0.5

    def test_articulation_values(self, detector, sample_embedding):
        """Test all articulation values are handled."""
        for articulation in ["smooth", "punchy", "staccato", "unknown"]:
            result = detector.analyze(
                rms_value=0.5,
                zcr_value=0.1,
                swing_ratio=0.6,
                articulation=articulation,
                bounciness=0.5,
                voice_probability=0.3,
                is_likely_instrumental=True,
                embedding=sample_embedding
            )

            assert 0.0 <= result['confidence_breakdown']['articulation'] <= 1.0
