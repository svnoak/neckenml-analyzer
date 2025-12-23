#!/usr/bin/env python3
"""
Classification Evaluation Script for neckenml-analyzer

This script evaluates the dancestyle classification accuracy by:
1. Loading test tracks from test_data/test_tracks.yaml
2. Running analysis and classification on each track
3. Comparing predictions vs ground truth
4. Generating detailed metrics and confusion matrix
5. Identifying critical misclassifications

Usage:
    python evaluate_classification.py
    python evaluate_classification.py --verbose
    python evaluate_classification.py --difficulty hard
    python evaluate_classification.py --style Polska
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import yaml
import json

# Add neckenml to path
try:
    from neckenml.analyzer import AudioAnalyzer
    from neckenml.classifier import StyleClassifier
except ImportError:
    print("Error: Cannot import neckenml. Make sure it's installed.")
    print("Try: pip install -e /path/to/neckenml")
    sys.exit(1)

import numpy as np
from datetime import datetime


class SimpleTrack:
    """Simple track object for classifier compatibility."""
    def __init__(self, title=None, artist=None):
        self.title = title
        self.artist = artist


class ClassificationEvaluator:
    """Evaluates classification accuracy on test dataset."""

    def __init__(self, test_data_path: str = "test_data/test_tracks.yaml"):
        self.test_data_path = Path(test_data_path)
        self.test_data = None
        self.results = []
        self.analyzer = AudioAnalyzer()
        self.classifier = StyleClassifier()

    def load_test_data(self) -> Dict:
        """Load test tracks from YAML file."""
        if not self.test_data_path.exists():
            raise FileNotFoundError(
                f"Test data file not found: {self.test_data_path}\n"
                f"Please create it using the provided template."
            )

        with open(self.test_data_path, 'r') as f:
            self.test_data = yaml.safe_load(f)

        print(f"Loaded {len(self.test_data.get('tracks', []))} test tracks")
        return self.test_data

    def analyze_track(self, track_info: Dict, verbose: bool = False) -> Dict:
        """
        Analyze a single track and return results.

        Returns:
            Dict with analysis results, classification, and comparison
        """
        file_path = Path(track_info['file_path'])

        if not file_path.exists():
            return {
                'track_id': track_info['id'],
                'error': 'File not found',
                'file_path': str(file_path),
                'status': 'skipped'
            }

        try:
            # Run analysis
            if verbose:
                print(f"  Analyzing {file_path.name}...")

            analysis_result = self.analyzer.analyze_file(str(file_path))

            # Extract features for classification
            # analyze_file returns features directly unless return_artifacts=True
            if 'features' in analysis_result:
                features = analysis_result['features']
            else:
                features = analysis_result

            # Run classification
            # Run classification (StyleClassifier needs track and analysis)
            track_obj = SimpleTrack(title=track_info.get("id", "Unknown"), artist=None)
            classification_results = self.classifier.classify(track_obj, features)
            classification = classification_results[0] if classification_results else {}

            # Extract key information
            predicted_style = classification.get('style', 'Unknown')
            predicted_substyle = classification.get('substyle')
            confidence = classification.get('confidence', 0.0)
            decision_path = classification.get('decision_path', 'unknown')  # metadata/ai/heuristic

            # Compare with ground truth
            true_style = track_info['true_style']
            is_correct = predicted_style == true_style

            # Determine error severity
            severity = self._get_error_severity(
                true_style,
                predicted_style,
                track_info.get('expected_meter')
            )

            result = {
                'track_id': track_info['id'],
                'file_path': str(file_path),
                'true_style': true_style,
                'true_substyle': track_info.get('substyle'),
                'predicted_style': predicted_style,
                'predicted_substyle': predicted_substyle,
                'confidence': confidence,
                'decision_path': decision_path,
                'is_correct': is_correct,
                'error_severity': severity if not is_correct else None,
                'difficulty': track_info.get('difficulty', 'medium'),
                'status': 'analyzed',
                'features': {
                    'bpm': features.get('tempo_bpm'),  # FIXED: analyzer returns 'tempo_bpm' not 'bpm'
                    'detected_meter': 'ternary' if features.get('ternary_confidence', 0) > 0.5 else 'binary',
                    'ternary_confidence': features.get('ternary_confidence'),
                    'polska_score': features.get('polska_score'),
                    'hambo_score': features.get('hambo_score'),
                    'swing_ratio': features.get('swing_ratio'),
                    'punchiness': features.get('punchiness'),
                },
                'notes': track_info.get('notes', ''),
            }

            if verbose and not is_correct:
                print(f"    ❌ WRONG: {true_style} → {predicted_style} "
                      f"(confidence: {confidence:.2f}, {decision_path})")
            elif verbose:
                print(f"    ✓ Correct: {predicted_style} (confidence: {confidence:.2f})")

            return result

        except Exception as e:
            return {
                'track_id': track_info['id'],
                'error': str(e),
                'file_path': str(file_path),
                'status': 'error'
            }

    def _get_error_severity(
        self,
        true_style: str,
        predicted_style: str,
        expected_meter: Optional[str]
    ) -> str:
        """
        Determine severity of misclassification.

        Returns:
            'critical' | 'high' | 'medium' | 'low'
        """
        # Define meter groups
        ternary_styles = {'Polska', 'Hambo', 'Vals', 'Slängpolska'}
        binary_styles = {'Polka', 'Schottis', 'Snoa', 'Engelska', 'Gånglåt', 'Marsch'}

        true_is_ternary = true_style in ternary_styles
        pred_is_ternary = predicted_style in ternary_styles

        # Critical: Wrong meter (would confuse dancers)
        if true_is_ternary != pred_is_ternary:
            return 'critical'

        # Critical pairs (very bad UX)
        critical_pairs = [
            {'Polska', 'Polka'},
            {'Hambo', 'Polka'},
        ]

        for pair in critical_pairs:
            if {true_style, predicted_style} == pair:
                return 'critical'

        # High severity
        high_severity_pairs = [
            {'Hambo', 'Polska'},
            {'Vals', 'Polka'},
        ]

        for pair in high_severity_pairs:
            if {true_style, predicted_style} == pair:
                return 'high'

        # Medium severity
        medium_pairs = [
            {'Schottis', 'Polka'},
            {'Snoa', 'Polka'},
            {'Vals', 'Polska'},
        ]

        for pair in medium_pairs:
            if {true_style, predicted_style} == pair:
                return 'medium'

        # Low severity (acceptable confusions)
        # Engelska/Polka, Gånglåt/Marsch, etc.
        return 'low'

    def evaluate_all(
        self,
        verbose: bool = False,
        filter_difficulty: Optional[str] = None,
        filter_style: Optional[str] = None
    ) -> List[Dict]:
        """
        Evaluate all test tracks.

        Args:
            verbose: Print detailed progress
            filter_difficulty: Only test tracks with this difficulty
            filter_style: Only test tracks of this style

        Returns:
            List of result dictionaries
        """
        if self.test_data is None:
            self.load_test_data()

        tracks = self.test_data.get('tracks', [])

        # Apply filters
        if filter_difficulty:
            tracks = [t for t in tracks if t.get('difficulty') == filter_difficulty]
            print(f"Filtering to difficulty={filter_difficulty}: {len(tracks)} tracks")

        if filter_style:
            tracks = [t for t in tracks if t.get('true_style') == filter_style]
            print(f"Filtering to style={filter_style}: {len(tracks)} tracks")

        if not tracks:
            print("No tracks to evaluate!")
            return []

        print(f"\n{'='*70}")
        print(f"Evaluating {len(tracks)} tracks...")
        print(f"{'='*70}\n")

        self.results = []

        for i, track in enumerate(tracks, 1):
            if verbose:
                print(f"[{i}/{len(tracks)}] {track['id']} - {track['true_style']}")

            result = self.analyze_track(track, verbose=verbose)
            self.results.append(result)

            if not verbose and i % 5 == 0:
                print(f"Progress: {i}/{len(tracks)} tracks analyzed...")

        print(f"\n{'='*70}")
        print("Evaluation complete!")
        print(f"{'='*70}\n")

        return self.results

    def generate_metrics(self) -> Dict:
        """Generate comprehensive metrics from results."""
        if not self.results:
            return {}

        # Filter out skipped/error tracks
        valid_results = [r for r in self.results if r.get('status') == 'analyzed']

        if not valid_results:
            print("No valid results to analyze!")
            return {}

        total = len(valid_results)
        correct = sum(1 for r in valid_results if r['is_correct'])

        # Overall accuracy
        overall_accuracy = correct / total if total > 0 else 0

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

        # Per-difficulty accuracy
        difficulty_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in valid_results:
            diff = result.get('difficulty', 'medium')
            difficulty_stats[diff]['total'] += 1
            if result['is_correct']:
                difficulty_stats[diff]['correct'] += 1

        per_difficulty_accuracy = {
            diff: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            for diff, stats in difficulty_stats.items()
        }

        # Decision path statistics
        decision_path_stats = Counter(r['decision_path'] for r in valid_results)
        decision_path_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in valid_results:
            path = result['decision_path']
            decision_path_accuracy[path]['total'] += 1
            if result['is_correct']:
                decision_path_accuracy[path]['correct'] += 1

        decision_path_acc = {
            path: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            for path, stats in decision_path_accuracy.items()
        }

        # Error analysis
        errors_by_severity = defaultdict(list)

        for result in valid_results:
            if not result['is_correct']:
                severity = result.get('error_severity', 'unknown')
                errors_by_severity[severity].append(result)

        # Confusion matrix data
        confusion_data = defaultdict(lambda: defaultdict(int))

        for result in valid_results:
            true_style = result['true_style']
            pred_style = result['predicted_style']
            confusion_data[true_style][pred_style] += 1

        return {
            'timestamp': datetime.now().isoformat(),
            'total_tracks': total,
            'correct_predictions': correct,
            'overall_accuracy': overall_accuracy,
            'per_style_accuracy': dict(per_style_accuracy),
            'per_style_counts': {k: v['total'] for k, v in style_stats.items()},
            'per_difficulty_accuracy': dict(per_difficulty_accuracy),
            'decision_path_distribution': dict(decision_path_stats),
            'decision_path_accuracy': dict(decision_path_acc),
            'errors_by_severity': {
                severity: len(errors)
                for severity, errors in errors_by_severity.items()
            },
            'confusion_matrix': {k: dict(v) for k, v in confusion_data.items()},
        }

    def print_report(self, metrics: Dict):
        """Print human-readable evaluation report."""
        print(f"\n{'='*70}")
        print("CLASSIFICATION EVALUATION REPORT")
        print(f"{'='*70}\n")

        print(f"Timestamp: {metrics['timestamp']}")
        print(f"Total Tracks: {metrics['total_tracks']}")
        print(f"Correct: {metrics['correct_predictions']}")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.1%}\n")

        # Per-style accuracy
        print(f"{'='*70}")
        print("PER-STYLE ACCURACY")
        print(f"{'='*70}")

        style_acc = metrics['per_style_accuracy']
        style_counts = metrics['per_style_counts']

        # Sort by count (most tested styles first)
        sorted_styles = sorted(
            style_acc.keys(),
            key=lambda s: style_counts[s],
            reverse=True
        )

        for style in sorted_styles:
            accuracy = style_acc[style]
            count = style_counts[style]
            bar = '█' * int(accuracy * 20)
            print(f"{style:15} {accuracy:6.1%} ({count:2} tracks) {bar}")

        # Per-difficulty accuracy
        if metrics['per_difficulty_accuracy']:
            print(f"\n{'='*70}")
            print("PER-DIFFICULTY ACCURACY")
            print(f"{'='*70}")

            for diff in ['easy', 'medium', 'hard']:
                if diff in metrics['per_difficulty_accuracy']:
                    accuracy = metrics['per_difficulty_accuracy'][diff]
                    bar = '█' * int(accuracy * 20)
                    print(f"{diff:10} {accuracy:6.1%} {bar}")

        # Decision path statistics
        print(f"\n{'='*70}")
        print("DECISION PATH ANALYSIS")
        print(f"{'='*70}")

        for path, count in metrics['decision_path_distribution'].items():
            accuracy = metrics['decision_path_accuracy'].get(path, 0)
            pct = count / metrics['total_tracks'] * 100
            print(f"{path:15} {count:3} tracks ({pct:5.1f}%) - Accuracy: {accuracy:.1%}")

        # Error severity
        if metrics['errors_by_severity']:
            print(f"\n{'='*70}")
            print("ERROR SEVERITY BREAKDOWN")
            print(f"{'='*70}")

            for severity in ['critical', 'high', 'medium', 'low']:
                if severity in metrics['errors_by_severity']:
                    count = metrics['errors_by_severity'][severity]
                    print(f"{severity.upper():10} {count:3} errors")

        # Detailed errors
        print(f"\n{'='*70}")
        print("MISCLASSIFICATIONS DETAILS")
        print(f"{'='*70}\n")

        errors = [r for r in self.results if not r.get('is_correct', False) and r.get('status') == 'analyzed']

        if not errors:
            print("🎉 No errors! Perfect classification!\n")
        else:
            # Group by severity
            for severity in ['critical', 'high', 'medium', 'low']:
                severity_errors = [e for e in errors if e.get('error_severity') == severity]

                if severity_errors:
                    print(f"\n{severity.upper()} ERRORS ({len(severity_errors)}):")
                    print("-" * 70)

                    for err in severity_errors:
                        print(f"  {err['track_id']}: {err['true_style']} → {err['predicted_style']}")
                        print(f"    Confidence: {err['confidence']:.2f} | Path: {err['decision_path']}")
                        print(f"    BPM: {err['features'].get('bpm', 'N/A')} | "
                              f"Meter: {err['features']['detected_meter']} "
                              f"(ternary conf: {err['features']['ternary_confidence']:.2f})")
                        if err.get('notes'):
                            print(f"    Notes: {err['notes']}")
                        print()

        print(f"{'='*70}\n")

    def save_results(self, output_path: str = "test_data/evaluation_results.json"):
        """Save detailed results to JSON file."""
        output = {
            'results': self.results,
            'metrics': self.generate_metrics(),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {output_path}")

    def get_confusion_matrix_data(self) -> Tuple[List[str], np.ndarray]:
        """
        Get confusion matrix in format suitable for visualization.

        Returns:
            (labels, matrix) where matrix is numpy array
        """
        metrics = self.generate_metrics()
        confusion_dict = metrics['confusion_matrix']

        # Get all unique labels
        all_labels = sorted(set(
            list(confusion_dict.keys()) +
            [pred for true_dict in confusion_dict.values() for pred in true_dict.keys()]
        ))

        # Build matrix
        n = len(all_labels)
        matrix = np.zeros((n, n), dtype=int)

        for i, true_label in enumerate(all_labels):
            for j, pred_label in enumerate(all_labels):
                if true_label in confusion_dict and pred_label in confusion_dict[true_label]:
                    matrix[i, j] = confusion_dict[true_label][pred_label]

        return all_labels, matrix


def main():
    parser = argparse.ArgumentParser(description='Evaluate dancestyle classification')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                       help='Filter by difficulty')
    parser.add_argument('--style', help='Filter by dance style')
    parser.add_argument('--output', '-o', default='test_data/evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate confusion matrix plot')

    args = parser.parse_args()

    # Create evaluator
    evaluator = ClassificationEvaluator()

    # Load and evaluate
    try:
        evaluator.load_test_data()
    except FileNotFoundError as e:
        print(f"\n❌ {e}\n")
        return 1

    # Run evaluation
    evaluator.evaluate_all(
        verbose=args.verbose,
        filter_difficulty=args.difficulty,
        filter_style=args.style
    )

    # Generate and print metrics
    metrics = evaluator.generate_metrics()
    evaluator.print_report(metrics)

    # Save results
    evaluator.save_results(args.output)

    # Generate confusion matrix plot if requested
    if args.plot:
        try:
            from visualize_confusion_matrix import plot_confusion_matrix
            labels, matrix = evaluator.get_confusion_matrix_data()
            plot_confusion_matrix(
                matrix,
                labels,
                save_path='test_data/confusion_matrix.png'
            )
            print("\nConfusion matrix plot saved to: test_data/confusion_matrix.png")
        except ImportError:
            print("\n⚠️  Cannot generate plot: visualize_confusion_matrix.py not found")

    return 0


if __name__ == '__main__':
    sys.exit(main())
