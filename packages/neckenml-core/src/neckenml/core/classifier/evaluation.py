"""
Held-out evaluation harness for the style classifier.

Grouping the cross-validation split by track, by album, or by artist
shows how much of a score comes from recognising the ensemble rather
than the dance style.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class EvaluationResult:
    """Pooled out-of-fold metrics from a grouped cross-validation run."""

    class_labels: list[str]
    precision: dict[str, float]
    recall: dict[str, float]
    confusion_matrix: object
    groups_per_class: dict[str, int]
    samples_per_class: dict[str, int]
    n_splits: int


def build_group_ids(ids: list) -> list[int]:
    """Map each id to a group number; a null id gets a group of its own."""
    group_ids = []
    id_to_group: dict[object, int] = {}
    next_group = 0
    for sample_id in ids:
        if sample_id is None:
            group_ids.append(next_group)
            next_group += 1
            continue
        if sample_id not in id_to_group:
            id_to_group[sample_id] = next_group
            next_group += 1
        group_ids.append(id_to_group[sample_id])
    return group_ids


def count_groups_per_class(labels: list, group_ids: list) -> dict[str, int]:
    """Count the distinct groups that hold each label."""
    groups_by_label: dict[str, set] = {}
    for label, group_id in zip(labels, group_ids):
        groups_by_label.setdefault(label, set()).add(group_id)
    return {label: len(groups) for label, groups in groups_by_label.items()}


def evaluate_classifier(embeddings, labels, group_ids: list, n_splits: int = 5) -> EvaluationResult:
    """Run grouped cross-validation and pool the out-of-fold predictions."""
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    group_ids = np.asarray(group_ids)

    splitter = StratifiedGroupKFold(n_splits=n_splits)

    true_labels = []
    predicted_labels = []

    for train_index, test_index in splitter.split(embeddings, labels, group_ids):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(embeddings[train_index])
        X_test = scaler.transform(embeddings[test_index])

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, labels[train_index])
        predictions = model.predict(X_test)

        true_labels.extend(labels[test_index])
        predicted_labels.extend(predictions)

    labels_list = labels.tolist()
    class_labels = sorted(set(labels_list))
    samples_per_class = {label: labels_list.count(label) for label in class_labels}

    precision_values, recall_values, _, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=class_labels, zero_division=0
    )
    matrix = sk_confusion_matrix(true_labels, predicted_labels, labels=class_labels)

    return EvaluationResult(
        class_labels=class_labels,
        precision=dict(zip(class_labels, precision_values.tolist())),
        recall=dict(zip(class_labels, recall_values.tolist())),
        confusion_matrix=matrix,
        groups_per_class=count_groups_per_class(labels.tolist(), group_ids.tolist()),
        samples_per_class=samples_per_class,
        n_splits=n_splits,
    )


def evaluate_by_grouping(
    embeddings,
    labels,
    track_ids: list,
    album_ids: list,
    artist_ids: list,
    n_splits: int = 5,
) -> dict[str, EvaluationResult]:
    """Evaluate the same data grouped by track, by album, and by artist."""
    return {
        "track": evaluate_classifier(
            embeddings, labels, build_group_ids(track_ids), n_splits=n_splits
        ),
        "album": evaluate_classifier(
            embeddings, labels, build_group_ids(album_ids), n_splits=n_splits
        ),
        "artist": evaluate_classifier(
            embeddings, labels, build_group_ids(artist_ids), n_splits=n_splits
        ),
    }
