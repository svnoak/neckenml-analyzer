#!/usr/bin/env python3
"""
Test that the parameterized classifier works correctly.
"""

from neckenml.classifier import StyleClassifier
from neckenml.classifier.params import ClassifierParams


def test_default_params():
    """Test classifier with default parameters."""
    print("Testing with default parameters...")

    classifier = StyleClassifier()

    # Check that params were loaded
    assert classifier.params is not None
    assert classifier.params.hambo_score_threshold == 0.30
    assert classifier.params.schottis_swing_threshold == 1.35

    print("✓ Default parameters loaded correctly")


def test_custom_params():
    """Test classifier with custom parameters."""
    print("\nTesting with custom parameters...")

    # Create custom params
    params = ClassifierParams()
    params.hambo_score_threshold = 0.25
    params.schottis_swing_threshold = 1.40

    # Create classifier with custom params
    classifier = StyleClassifier(params=params)

    # Verify custom params were used
    assert classifier.params.hambo_score_threshold == 0.25
    assert classifier.params.schottis_swing_threshold == 1.40

    print("✓ Custom parameters applied correctly")


def test_save_load_params():
    """Test saving and loading parameters."""
    print("\nTesting save/load parameters...")

    import tempfile
    import os

    # Create custom params
    params = ClassifierParams()
    params.hambo_score_threshold = 0.27
    params.version = 2
    params.training_samples = 100
    params.accuracy_on_training = 0.85

    # Save to temp file
    temp_file = tempfile.mktemp(suffix='.json')

    try:
        params.save(temp_file)
        print(f"  Saved to {temp_file}")

        # Load back
        loaded_params = ClassifierParams.load(temp_file)

        # Verify
        assert loaded_params.hambo_score_threshold == 0.27
        assert loaded_params.version == 2
        assert loaded_params.training_samples == 100
        assert loaded_params.accuracy_on_training == 0.85

        print("✓ Save/load works correctly")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_classification_still_works():
    """Test that classification actually works with params."""
    print("\nTesting classification functionality...")

    classifier = StyleClassifier()

    # Create sample analysis data
    analysis = {
        'tempo_bpm': 120,
        'meter': '3/4',
        'swing_ratio': 1.5,
        'avg_beat_ratios': [0.35, 0.32, 0.33],
        'punchiness': 0.5,
        'polska_score': 0.3,
        'hambo_score': 0.45,
        'ternary_confidence': 0.9,
        'bars': [],
        'sections': [],
        'embedding': None,
    }

    class SimpleTrack:
        def __init__(self):
            self.title = "Test Polska"
            self.artist = "Test Artist"

    track = SimpleTrack()

    # Run classification
    try:
        results = classifier.classify(track, analysis)
        assert len(results) > 0
        assert 'style' in results[0]

        print(f"  Classified as: {results[0]['style']} ({results[0]['confidence']:.2f})")
        print("✓ Classification works correctly")

    except Exception as e:
        print(f"✗ Classification failed: {e}")
        raise


if __name__ == '__main__':
    print("="*70)
    print("TESTING PARAMETERIZED CLASSIFIER")
    print("="*70)

    try:
        test_default_params()
        test_custom_params()
        test_save_load_params()
        test_classification_still_works()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
