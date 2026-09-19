"""
Tests for the held-out evaluation harness (evaluate_classifier and friends).

Tests cover:
- Grouping samples by a shared id (build_group_ids)
- Counting distinct groups per label (count_groups_per_class)
- Running grouped cross-validation over the style classifier (evaluate_classifier)
- Comparing track/album/artist grouping side by side (evaluate_by_grouping)
"""

import pytest
import numpy as np
from pathlib import Path

from neckenml.core.classifier.evaluation import (
    build_group_ids,
    count_groups_per_class,
    evaluate_classifier,
    evaluate_by_grouping,
    EvaluationResult,
)
from neckenml.core.classifier.style_head import ClassificationHead


def _separable_dataset(n_per_class=25, seed=42):
    """A small, deterministic, linearly separable two-class dataset."""
    rng = np.random.default_rng(seed)
    class_a = rng.normal(loc=0.0, scale=0.5, size=(n_per_class, 217))
    class_b = rng.normal(loc=6.0, scale=0.5, size=(n_per_class, 217))
    embeddings = np.vstack([class_a, class_b])
    labels = ["Polska"] * n_per_class + ["Hambo"] * n_per_class
    group_ids = list(range(len(labels)))
    return embeddings, labels, group_ids


def _one_group_dataset():
    """A dataset where every 'Polska' sample shares a single group id."""
    rng = np.random.default_rng(7)
    class_a = rng.normal(loc=0.0, scale=0.3, size=(6, 217))
    class_b = rng.normal(loc=6.0, scale=0.3, size=(6, 217))
    embeddings = np.vstack([class_a, class_b])
    labels = ["Polska"] * 6 + ["Hambo"] * 6
    group_ids = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    return embeddings, labels, group_ids


class TestBuildGroupIds:
    """Test suite for build_group_ids."""

    def test_samples_sharing_an_id_share_a_group(self):
        """Two samples with the same non-null id hold the same group number."""
        group_ids = build_group_ids(["albumA", "albumA", "albumB"])

        assert group_ids[0] == group_ids[1]

    def test_samples_with_different_ids_hold_different_groups(self):
        """Three samples with three distinct ids hold three distinct groups."""
        group_ids = build_group_ids(["albumA", "albumB", "albumC"])

        assert len(set(group_ids)) == 3

    def test_a_null_id_holds_a_group_of_its_own(self):
        """Two samples that both hold a null id never share a group."""
        group_ids = build_group_ids([None, None])

        assert group_ids[0] != group_ids[1]


class TestCountGroupsPerClass:
    """Test suite for count_groups_per_class."""

    def test_counts_the_distinct_groups_that_hold_a_label(self):
        """A label spread over three distinct groups counts three."""
        labels = ["Polska", "Polska", "Polska", "Hambo"]
        group_ids = [0, 1, 2, 3]

        counts = count_groups_per_class(labels, group_ids)

        assert counts["Polska"] == 3

    def test_a_label_inside_one_group_counts_one(self):
        """A label whose samples all share one group counts one."""
        labels = ["Polska", "Polska"]
        group_ids = [5, 5]

        counts = count_groups_per_class(labels, group_ids)

        assert counts["Polska"] == 1


class TestEvaluateClassifier:
    """Test suite for evaluate_classifier."""

    def test_reports_precision_and_recall_for_each_class(self):
        """Precision and recall hold one entry for each class, each in [0, 1]."""
        embeddings, labels, group_ids = _separable_dataset()

        result = evaluate_classifier(embeddings, labels, group_ids, n_splits=5)

        assert set(result.precision.keys()) == set(result.class_labels)
        assert set(result.recall.keys()) == set(result.class_labels)
        for value in list(result.precision.values()) + list(result.recall.values()):
            assert 0.0 <= value <= 1.0

    def test_the_confusion_matrix_is_square_over_the_class_labels(self):
        """The confusion matrix side equals the number of class labels."""
        embeddings, labels, group_ids = _separable_dataset()

        result = evaluate_classifier(embeddings, labels, group_ids, n_splits=5)

        matrix = np.array(result.confusion_matrix)
        n_classes = len(result.class_labels)
        assert matrix.shape == (n_classes, n_classes)

    def test_reports_the_group_count_for_a_class_inside_one_group(self):
        """A class whose samples all share one group counts one group."""
        embeddings, labels, group_ids = _one_group_dataset()

        result = evaluate_classifier(embeddings, labels, group_ids, n_splits=3)

        assert result.groups_per_class["Polska"] == 1

    def test_reports_the_sample_count_for_each_class(self):
        """samples_per_class holds the true sample count for each class."""
        rng = np.random.default_rng(3)
        class_a = rng.normal(loc=0.0, scale=0.5, size=(9, 217))
        class_b = rng.normal(loc=6.0, scale=0.5, size=(15, 217))
        embeddings = np.vstack([class_a, class_b])
        labels = ["Polska"] * 9 + ["Hambo"] * 15
        group_ids = list(range(len(labels)))

        result = evaluate_classifier(embeddings, labels, group_ids, n_splits=3)

        assert result.samples_per_class["Polska"] == 9
        assert result.samples_per_class["Hambo"] == 15
        assert set(result.samples_per_class.keys()) == set(result.class_labels)

    def test_drops_a_vector_of_the_wrong_length(self):
        """A vector whose length differs from the expected feature count is dropped."""
        embeddings, labels, group_ids = _separable_dataset()
        embeddings = list(embeddings) + [np.zeros(10)]
        labels = labels + ["Polska"]
        group_ids = group_ids + [max(group_ids) + 1]

        result = evaluate_classifier(embeddings, labels, group_ids, n_splits=5)

        assert sum(result.samples_per_class.values()) == len(labels) - 1

    def test_raises_when_no_sample_has_the_expected_feature_count(self):
        """The harness raises rather than fit a model on zero samples."""
        rng = np.random.default_rng(11)
        class_a = rng.normal(loc=0.0, scale=0.5, size=(25, 10))
        class_b = rng.normal(loc=6.0, scale=0.5, size=(25, 10))
        embeddings = np.vstack([class_a, class_b])
        labels = ["Polska"] * 25 + ["Hambo"] * 25
        group_ids = list(range(len(labels)))

        with pytest.raises(ValueError):
            evaluate_classifier(embeddings, labels, group_ids, n_splits=5)

    def test_precision_and_recall_follow_the_pooled_confusion_matrix(self):
        """Precision and recall come from the pooled confusion matrix, not a per-fold average."""
        rng = np.random.default_rng(5)
        class_a = rng.normal(loc=0.0, scale=2.5, size=(30, 217))
        class_b = rng.normal(loc=2.0, scale=2.5, size=(30, 217))
        class_c = rng.normal(loc=4.0, scale=2.5, size=(30, 217))
        embeddings = np.vstack([class_a, class_b, class_c])
        labels = ["Polska"] * 30 + ["Hambo"] * 30 + ["Schottis"] * 30
        group_ids = list(range(len(labels)))

        result = evaluate_classifier(embeddings, labels, group_ids, n_splits=5)

        matrix = np.array(result.confusion_matrix)
        assert not np.array_equal(matrix, np.diag(np.diag(matrix)))

        for i, label in enumerate(result.class_labels):
            column_sum = matrix[:, i].sum()
            row_sum = matrix[i, :].sum()
            expected_precision = matrix[i, i] / column_sum if column_sum else 0.0
            expected_recall = matrix[i, i] / row_sum if row_sum else 0.0

            assert result.precision[label] == pytest.approx(expected_precision)
            assert result.recall[label] == pytest.approx(expected_recall)

    def test_the_harness_writes_no_file(self, tmp_path, monkeypatch):
        """The harness fits and predicts in memory; it constructs no model file."""

        def _fail_init(self, model_path=None):
            raise AssertionError("the evaluation harness must not construct a ClassificationHead")

        monkeypatch.setattr(ClassificationHead, "__init__", _fail_init)

        model_dir = Path.home() / ".neckenml"
        before_tmp = set(tmp_path.rglob("*"))
        before_model_dir = set(model_dir.rglob("*")) if model_dir.exists() else set()

        embeddings, labels, group_ids = _one_group_dataset()
        evaluate_classifier(embeddings, labels, group_ids, n_splits=3)

        after_tmp = set(tmp_path.rglob("*"))
        after_model_dir = set(model_dir.rglob("*")) if model_dir.exists() else set()

        assert after_tmp == before_tmp
        assert after_model_dir == before_model_dir


class TestEvaluateByGrouping:
    """Test suite for evaluate_by_grouping."""

    def test_returns_a_result_for_track_album_and_artist(self):
        """The returned dictionary holds exactly the track, album, and artist keys."""
        embeddings, labels, _ = _separable_dataset()
        n = len(labels)
        track_ids = [f"track-{i}" for i in range(n)]
        album_ids = [f"album-{i}" for i in range(n)]
        artist_ids = [f"artist-{i}" for i in range(n)]

        results = evaluate_by_grouping(embeddings, labels, track_ids, album_ids, artist_ids)

        assert set(results.keys()) == {"track", "album", "artist"}

    def test_an_absent_album_does_not_join_two_samples(self):
        """Two samples with no album id never share an album group."""
        embeddings, labels, _ = _one_group_dataset()
        n = len(labels)
        track_ids = [f"track-{i}" for i in range(n)]
        artist_ids = [f"artist-{i}" for i in range(n)]
        album_ids = [None, None] + [f"album-{i}" for i in range(2, n)]

        results = evaluate_by_grouping(
            embeddings, labels, track_ids, album_ids, artist_ids, n_splits=3
        )

        assert results["album"].groups_per_class["Polska"] == 6
