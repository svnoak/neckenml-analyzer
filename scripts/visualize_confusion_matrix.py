#!/usr/bin/env python3
"""
Confusion Matrix Visualization for Classification Results

This script creates publication-quality confusion matrix visualizations
with additional analysis features.

Usage:
    python visualize_confusion_matrix.py
    python visualize_confusion_matrix.py --results test_data/evaluation_results.json
    python visualize_confusion_matrix.py --normalize --annotate
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    normalize: bool = False,
    annotate: bool = True,
    title: str = "Dance Style Classification Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    cmap: str = 'Blues'
):
    """
    Plot a confusion matrix with advanced visualization options.

    Args:
        matrix: Confusion matrix (rows=true, cols=predicted)
        labels: List of class labels
        normalize: If True, normalize by row (true class)
        annotate: If True, show values in cells
        title: Plot title
        save_path: Path to save figure (if None, displays interactively)
        figsize: Figure size
        cmap: Colormap name
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize if requested
    if normalize:
        # Normalize by row (true class) - shows recall
        matrix_display = matrix.astype('float') / matrix.sum(axis=1, keepdims=True)
        matrix_display = np.nan_to_num(matrix_display)  # Handle division by zero
        fmt = '.2f'
        cbar_label = 'Proportion'
    else:
        matrix_display = matrix
        fmt = 'd'
        cbar_label = 'Count'

    # Create heatmap
    im = ax.imshow(matrix_display, interpolation='nearest', cmap=cmap)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add annotations
    if annotate:
        # Determine threshold for text color (light vs dark)
        thresh = matrix_display.max() / 2.

        for i in range(len(labels)):
            for j in range(len(labels)):
                # Show both count and percentage if normalized
                if normalize:
                    text_value = f"{matrix_display[i, j]:.2f}"
                    if matrix[i, j] > 0:
                        text_value += f"\n({matrix[i, j]})"
                else:
                    text_value = f"{matrix[i, j]}"

                # Color text based on background
                text_color = "white" if matrix_display[i, j] > thresh else "black"

                # Highlight diagonal (correct predictions) in bold
                weight = 'bold' if i == j else 'normal'

                ax.text(j, i, text_value,
                       ha="center", va="center",
                       color=text_color,
                       fontsize=9,
                       weight=weight)

    # Labels
    ax.set_ylabel('True Dance Style', fontsize=12, weight='bold')
    ax.set_xlabel('Predicted Dance Style', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)

    # Add grid
    ax.set_xticks(np.arange(len(labels))-.5, minor=True)
    ax.set_yticks(np.arange(len(labels))-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Tight layout
    fig.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_confusion_with_insights(
    matrix: np.ndarray,
    labels: List[str],
    save_dir: str = "test_data",
):
    """
    Create multiple visualization perspectives of the confusion matrix.

    Generates:
    1. Standard confusion matrix (counts)
    2. Normalized confusion matrix (percentages)
    3. Error highlights (off-diagonal only)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard confusion matrix with counts
    print("\nGenerating standard confusion matrix...")
    plot_confusion_matrix(
        matrix,
        labels,
        normalize=False,
        annotate=True,
        title="Dance Style Classification - Counts",
        save_path=save_dir / "confusion_matrix_counts.png"
    )
    plt.close()

    # 2. Normalized confusion matrix (recall per class)
    print("Generating normalized confusion matrix...")
    plot_confusion_matrix(
        matrix,
        labels,
        normalize=True,
        annotate=True,
        title="Dance Style Classification - Recall (Normalized by True Class)",
        save_path=save_dir / "confusion_matrix_normalized.png"
    )
    plt.close()

    # 3. Error-focused view (only misclassifications)
    print("Generating error-focused matrix...")
    error_matrix = matrix.copy()
    np.fill_diagonal(error_matrix, 0)  # Remove correct predictions

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use red colormap for errors
    im = ax.imshow(error_matrix, interpolation='nearest', cmap='Reds')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Error Count', rotation=-90, va="bottom", fontsize=11)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate errors
    thresh = error_matrix.max() / 2. if error_matrix.max() > 0 else 1

    for i in range(len(labels)):
        for j in range(len(labels)):
            if error_matrix[i, j] > 0:
                text_color = "white" if error_matrix[i, j] > thresh else "black"
                ax.text(j, i, f"{error_matrix[i, j]}",
                       ha="center", va="center",
                       color=text_color,
                       fontsize=10,
                       weight='bold')

    ax.set_ylabel('True Dance Style', fontsize=12, weight='bold')
    ax.set_xlabel('Predicted Dance Style', fontsize=12, weight='bold')
    ax.set_title('Misclassifications Only (Correct Predictions Hidden)',
                fontsize=14, weight='bold', pad=20)

    ax.set_xticks(np.arange(len(labels))-.5, minor=True)
    ax.set_yticks(np.arange(len(labels))-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    fig.tight_layout()
    plt.savefig(save_dir / "confusion_matrix_errors_only.png", dpi=300, bbox_inches='tight')
    print(f"Error matrix saved to: {save_dir / 'confusion_matrix_errors_only.png'}")
    plt.close()

    # 4. Per-class accuracy bar chart
    print("Generating per-class accuracy chart...")
    plot_per_class_accuracy(matrix, labels, save_dir / "per_class_accuracy.png")


def plot_per_class_accuracy(
    matrix: np.ndarray,
    labels: List[str],
    save_path: str
):
    """Plot per-class accuracy as horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate accuracy per class
    accuracies = []
    totals = []

    for i in range(len(labels)):
        total = matrix[i, :].sum()
        correct = matrix[i, i]
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)
        totals.append(total)

    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    sorted_totals = [totals[i] for i in sorted_indices]

    # Color bars by accuracy level
    colors = []
    for acc in sorted_accuracies:
        if acc >= 0.9:
            colors.append('#2ecc71')  # Green
        elif acc >= 0.7:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red

    # Create horizontal bar chart
    y_pos = np.arange(len(sorted_labels))
    bars = ax.barh(y_pos, sorted_accuracies, color=colors, alpha=0.8)

    # Add percentage labels
    for i, (bar, acc, total) in enumerate(zip(bars, sorted_accuracies, sorted_totals)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
               f'{acc:.1%} ({total} tracks)',
               ha='left', va='center', fontsize=9)

    # Add reference line at 90%
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='90% target')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.set_xlabel('Accuracy (Recall)', fontsize=12, weight='bold')
    ax.set_title('Classification Accuracy by Dance Style', fontsize=14, weight='bold', pad=20)
    ax.set_xlim([0, 1.1])
    ax.legend(loc='lower right')

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class accuracy chart saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize confusion matrix from evaluation results'
    )
    parser.add_argument(
        '--results',
        default='test_data/evaluation_results.json',
        help='Path to evaluation results JSON'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize confusion matrix by row'
    )
    parser.add_argument(
        '--no-annotate',
        action='store_true',
        help='Do not show values in cells'
    )
    parser.add_argument(
        '--output-dir',
        default='test_data',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Generate only single confusion matrix (not full suite)'
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        print("\nRun evaluate_classification.py first to generate results.")
        return 1

    with open(results_path, 'r') as f:
        data = json.load(f)

    metrics = data.get('metrics', {})
    confusion_dict = metrics.get('confusion_matrix', {})

    if not confusion_dict:
        print("❌ No confusion matrix data found in results.")
        return 1

    # Build matrix
    all_labels = sorted(set(
        list(confusion_dict.keys()) +
        [pred for true_dict in confusion_dict.values() for pred in true_dict.keys()]
    ))

    n = len(all_labels)
    matrix = np.zeros((n, n), dtype=int)

    for i, true_label in enumerate(all_labels):
        for j, pred_label in enumerate(all_labels):
            if true_label in confusion_dict and pred_label in confusion_dict[true_label]:
                matrix[i, j] = confusion_dict[true_label][pred_label]

    print(f"\nLoaded confusion matrix: {n}x{n} ({len(all_labels)} dance styles)")
    print(f"Total predictions: {matrix.sum()}")

    # Generate visualizations
    if args.simple:
        # Single plot
        save_path = Path(args.output_dir) / "confusion_matrix.png"
        plot_confusion_matrix(
            matrix,
            all_labels,
            normalize=args.normalize,
            annotate=not args.no_annotate,
            save_path=str(save_path)
        )
    else:
        # Full suite
        print("\nGenerating full visualization suite...")
        plot_confusion_with_insights(matrix, all_labels, args.output_dir)
        print(f"\n✅ All visualizations saved to: {args.output_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
