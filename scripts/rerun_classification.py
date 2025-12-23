#!/usr/bin/env python3
"""
Re-run classification on existing evaluation results.
This script takes the old evaluation_results.json and re-classifies
using the updated classifier logic without re-analyzing audio.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add neckenml to path
try:
    from neckenml.classifier import StyleClassifier
except ImportError:
    print("Error: Cannot import neckenml. Make sure it's installed.")
    sys.exit(1)


class SimpleTrack:
    """Simple track object for classifier compatibility."""
    def __init__(self, title=None):
        self.title = title


def rerun_classification(input_file: str, output_file: str):
    """Re-classify existing evaluation results."""

    # Load previous results
    with open(input_file, 'r') as f:
        data = json.load(f)

    old_results = data['results']
    print(f"Loaded {len(old_results)} previous results")

    # Initialize classifier
    classifier = StyleClassifier()

    # Re-classify each track
    new_results = []
    correct_count = 0

    for i, old_result in enumerate(old_results, 1):
        if old_result.get('status') != 'analyzed':
            new_results.append(old_result)
            continue

        # Extract features from old result
        features = old_result['features']

        # Add missing fields that classifier expects (with defaults)
        analysis = {
            'tempo_bpm': features.get('bpm', 0) or 0,
            'meter': '3/4' if features.get('detected_meter') == 'ternary' else '4/4',
            'swing_ratio': features.get('swing_ratio', 1.0),
            'avg_beat_ratios': [0.33, 0.33, 0.33],  # Default
            'punchiness': features.get('punchiness', 0),
            'polska_score': features.get('polska_score', 0),
            'hambo_score': features.get('hambo_score', 0),
            'ternary_confidence': features.get('ternary_confidence', 0.5),
            'bars': [],
            'sections': [],
            'embedding': None,  # No ML classification
        }

        # Create simple track object
        track = SimpleTrack(title=old_result['track_id'])

        # Run classification
        try:
            classification_results = classifier.classify(track, analysis)
            classification = classification_results[0] if classification_results else {}

            predicted_style = classification.get('style', 'Unknown')
            confidence = classification.get('confidence', 0.0)
            source = classification.get('source', 'unknown')

            # Check if correct
            true_style = old_result['true_style']
            is_correct = predicted_style == true_style

            if is_correct:
                correct_count += 1

            # Determine error severity
            severity = get_error_severity(true_style, predicted_style) if not is_correct else None

            # Build new result
            new_result = {
                **old_result,
                'predicted_style': predicted_style,
                'confidence': confidence,
                'decision_path': source,
                'is_correct': is_correct,
                'error_severity': severity,
            }

            new_results.append(new_result)

            # Progress
            if i % 10 == 0:
                print(f"  Processed {i}/{len(old_results)}...")

        except Exception as e:
            print(f"Error classifying {old_result['track_id']}: {e}")
            new_results.append(old_result)

    # Calculate new metrics
    total = len([r for r in new_results if r.get('status') == 'analyzed'])
    accuracy = correct_count / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"Total tracks: {total}")
    print(f"Correct: {correct_count}")
    print(f"Overall Accuracy: {accuracy:.1%}")

    # Calculate per-style accuracy
    style_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for result in new_results:
        if result.get('status') != 'analyzed':
            continue
        true_style = result['true_style']
        style_stats[true_style]['total'] += 1
        if result['is_correct']:
            style_stats[true_style]['correct'] += 1

    print(f"\nPer-Style Accuracy:")
    for style in sorted(style_stats.keys()):
        stats = style_stats[style]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {style:15} {acc:6.1%} ({stats['correct']}/{stats['total']})")

    # Save new results
    output_data = {
        'results': new_results,
        'metrics': generate_metrics(new_results)
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nNew results saved to: {output_file}")

    # Show improvement comparison
    old_accuracy = data['metrics']['overall_accuracy']
    improvement = accuracy - old_accuracy
    print(f"\n{'='*70}")
    print(f"IMPROVEMENT")
    print(f"{'='*70}")
    print(f"Old Accuracy: {old_accuracy:.1%}")
    print(f"New Accuracy: {accuracy:.1%}")
    print(f"Improvement:  {improvement:+.1%}")
    print(f"{'='*70}\n")


def get_error_severity(true_style: str, predicted_style: str) -> str:
    """Determine severity of misclassification."""
    ternary_styles = {'Polska', 'Hambo', 'Vals', 'Slängpolska', 'Mazurka', 'Menuett'}
    binary_styles = {'Polka', 'Schottis', 'Snoa', 'Engelska'}

    true_is_ternary = true_style in ternary_styles
    pred_is_ternary = predicted_style in ternary_styles

    # Critical: Wrong meter
    if true_is_ternary != pred_is_ternary:
        return 'critical'

    # Critical pairs
    if {true_style, predicted_style} in [{'Polska', 'Polka'}, {'Hambo', 'Polka'}]:
        return 'critical'

    # High severity
    if {true_style, predicted_style} in [{'Hambo', 'Polska'}, {'Vals', 'Polka'}]:
        return 'high'

    # Medium severity
    if {true_style, predicted_style} in [{'Schottis', 'Polka'}, {'Snoa', 'Polka'}, {'Vals', 'Polska'}]:
        return 'medium'

    return 'low'


def generate_metrics(results):
    """Generate metrics from results."""
    from datetime import datetime

    valid_results = [r for r in results if r.get('status') == 'analyzed']
    total = len(valid_results)
    correct = sum(1 for r in valid_results if r['is_correct'])

    # Per-style accuracy
    style_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for result in valid_results:
        true_style = result['true_style']
        style_stats[true_style]['total'] += 1
        if result['is_correct']:
            style_stats[true_style]['correct'] += 1

    per_style_accuracy = {
        style: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        for style, stats in style_stats.items()
    }

    # Confusion matrix
    confusion_data = defaultdict(lambda: defaultdict(int))
    for result in valid_results:
        true_style = result['true_style']
        pred_style = result['predicted_style']
        confusion_data[true_style][pred_style] += 1

    # Error severity
    errors_by_severity = defaultdict(int)
    for result in valid_results:
        if not result['is_correct']:
            severity = result.get('error_severity', 'unknown')
            errors_by_severity[severity] += 1

    return {
        'timestamp': datetime.now().isoformat(),
        'total_tracks': total,
        'correct_predictions': correct,
        'overall_accuracy': correct / total if total > 0 else 0,
        'per_style_accuracy': dict(per_style_accuracy),
        'per_style_counts': {k: v['total'] for k, v in style_stats.items()},
        'confusion_matrix': {k: dict(v) for k, v in confusion_data.items()},
        'errors_by_severity': dict(errors_by_severity),
    }


if __name__ == '__main__':
    input_file = 'test_data/evaluation_results.json'
    output_file = 'test_data/evaluation_results_adjusted.json'

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found!")
        sys.exit(1)

    print("Re-running classification with adjusted classifier...\n")
    rerun_classification(input_file, output_file)
