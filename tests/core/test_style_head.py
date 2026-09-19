"""
Tests for the ClassificationHead classifier.

Tests cover:
- Training on validated feature vectors only
"""
import numpy as np
from neckenml.core import ClassificationHead


class TestTrain:
    """Test suite for ClassificationHead.train."""

    def test_train_fits_only_the_validated_vectors(self, tmp_path):
        """Test that train fits on vectors it validated, dropping the wrong-length one."""
        np.random.seed(42)

        model_path = str(tmp_path / "custom_style_head.pkl")
        head = ClassificationHead(model_path=model_path)

        embeddings = []
        labels = []

        for _ in range(5):
            embeddings.append(np.random.randn(ClassificationHead.EXPECTED_FEATURE_COUNT).tolist())
            labels.append("Polska")

        for _ in range(5):
            embeddings.append(np.random.randn(ClassificationHead.EXPECTED_FEATURE_COUNT).tolist())
            labels.append("Vals")

        embeddings.append(np.random.randn(ClassificationHead.EXPECTED_FEATURE_COUNT + 1).tolist())
        labels.append("Menuett")

        head.train(embeddings, labels)

        classes = head.model.classes_.tolist()
        assert "Polska" in classes
        assert "Vals" in classes
        assert "Menuett" not in classes
        assert head.scaler.n_features_in_ == ClassificationHead.EXPECTED_FEATURE_COUNT
