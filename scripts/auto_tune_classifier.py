#!/usr/bin/env python3
"""
Auto-Tuning Classifier - Automatically optimizes classification thresholds

This script uses Bayesian optimization to find the best classifier thresholds
by testing different parameter combinations against your ground truth data.

Usage:
    python3 auto_tune_classifier.py
    python3 auto_tune_classifier.py --target 0.90  # Target 90% accuracy
    python3 auto_tune_classifier.py --max-iterations 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class ClassifierParams:
    """Tunable parameters for the classifier."""
    # Hambo detection
    hambo_score_threshold: float = 0.30
    hambo_ratio_threshold: float = 0.40

    # Schottis detection
    schottis_swing_threshold: float = 1.35

    # Vals detection
    vals_ratio_tolerance: float = 0.08
    vals_polska_score_max: float = 0.30

    # Polska detection
    polska_score_fallback: float = 0.40
    polska_rescue_ternary_min: float = 0.55
    polska_rescue_polska_score_min: float = 0.25
    polska_rescue_signals_required: int = 4

    # Mazurka detection
    mazurka_swing_high: float = 1.80
    mazurka_swing_medium: float = 1.20

    # Menuett detection
    menuett_swing_min: float = 0.70
    menuett_swing_max: float = 1.05
    menuett_bpm_max: float = 115.0

    # Engelska/Polka detection
    engelska_swing_min: float = 0.70
    engelska_swing_max: float = 0.92
    engelska_bpm_min: float = 115.0
    polka_swing_max: float = 1.10

    # Snoa detection
    snoa_bpm_min: float = 80.0
    snoa_bpm_max: float = 115.0
    snoa_swing_max: float = 1.20
    snoa_swing_fallback_min: float = 0.90
    snoa_swing_fallback_max: float = 1.15


class SimpleTrack:
    """Simple track object for classifier compatibility."""
    def __init__(self, title=None):
        self.title = title


class AutoTuningClassifier:
    """Self-tuning classifier that optimizes thresholds."""

    def __init__(
        self,
        evaluation_data_path: str = "test_data/evaluation_results.json",
        test_split: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize auto-tuning classifier.

        Args:
            evaluation_data_path: Path to evaluation results JSON
            test_split: Fraction of data to hold out for testing (0.0-0.5)
            random_seed: Random seed for reproducible splits
        """
        self.eval_data_path = Path(evaluation_data_path)
        self.test_split = test_split
        self.random_seed = random_seed

        # Load and split data
        all_data = self._load_ground_truth()
        self.train_data, self.test_data = self._split_data(all_data)

        print(f"Data split: {len(self.train_data)} train, {len(self.test_data)} test")

        # Import here to avoid circular dependencies
        from neckenml.classifier import StyleClassifier
        self.classifier = StyleClassifier()

    def _load_ground_truth(self) -> List[Dict]:
        """Load ground truth data from evaluation results."""
        if not self.eval_data_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {self.eval_data_path}")

        with open(self.eval_data_path, 'r') as f:
            data = json.load(f)

        # Extract only analyzed tracks
        ground_truth = [
            r for r in data['results']
            if r.get('status') == 'analyzed' and r.get('features')
        ]

        print(f"Loaded {len(ground_truth)} total tracks")
        return ground_truth

    def _split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Split data into train/test sets with stratification.

        Ensures each style is represented in both train and test sets.
        """
        import random
        from collections import defaultdict

        # Set random seed for reproducibility
        random.seed(self.random_seed)

        # Group by style for stratified split
        by_style = defaultdict(list)
        for item in data:
            style = item['true_style']
            by_style[style].append(item)

        train_data = []
        test_data = []

        # Split each style separately
        for style, items in by_style.items():
            # Shuffle items
            shuffled = items.copy()
            random.shuffle(shuffled)

            # Calculate split point
            n_test = max(1, int(len(items) * self.test_split))  # At least 1 test sample
            n_train = len(items) - n_test

            # If we only have 1-2 items, put all in train (can't split)
            if len(items) <= 2:
                train_data.extend(shuffled)
                print(f"  {style}: {len(items)} train, 0 test (too few samples)")
            else:
                train_data.extend(shuffled[:n_train])
                test_data.extend(shuffled[n_train:])
                print(f"  {style}: {n_train} train, {n_test} test")

        return train_data, test_data

    def _create_analysis_dict(self, features: Dict, params: ClassifierParams) -> Dict:
        """Create analysis dict from features for classification."""
        return {
            'tempo_bpm': features.get('bpm', 0) or 0,
            'meter': '3/4' if features.get('detected_meter') == 'ternary' else '4/4',
            'swing_ratio': features.get('swing_ratio', 1.0),
            'avg_beat_ratios': [0.33, 0.33, 0.33],  # Default, would be better if stored
            'punchiness': features.get('punchiness', 0),
            'polska_score': features.get('polska_score', 0),
            'hambo_score': features.get('hambo_score', 0),
            'ternary_confidence': features.get('ternary_confidence', 0.5),
            'bars': [],
            'sections': [],
            'embedding': None,  # No ML classification in optimization
        }

    def evaluate_params(self, params: ClassifierParams, dataset: List[Dict] = None, verbose: bool = False) -> Tuple[float, Dict]:
        """
        Evaluate classifier with given parameters on specified dataset.

        Args:
            params: ClassifierParams to test
            dataset: Dataset to evaluate on (defaults to train_data)
            verbose: Print detailed progress

        Returns:
            (accuracy, per_style_metrics)
        """
        # Use train_data if no dataset specified
        if dataset is None:
            dataset = self.train_data

        # Apply params to classifier
        self._apply_params_to_classifier(params)

        correct = 0
        total = 0
        per_style_correct = {}
        per_style_total = {}

        for result in dataset:
            features = result['features']
            true_style = result['true_style']

            # Track per-style stats
            if true_style not in per_style_total:
                per_style_total[true_style] = 0
                per_style_correct[true_style] = 0
            per_style_total[true_style] += 1

            # Create analysis dict
            analysis = self._create_analysis_dict(features, params)

            # Classify
            track = SimpleTrack(title=result['track_id'])
            try:
                classifications = self.classifier.classify(track, analysis)
                predicted_style = classifications[0]['style'] if classifications else 'Unknown'

                if predicted_style == true_style:
                    correct += 1
                    per_style_correct[true_style] += 1

                total += 1

            except Exception as e:
                if verbose:
                    print(f"Error classifying {result['track_id']}: {e}")
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        # Calculate per-style accuracy
        per_style_acc = {
            style: per_style_correct[style] / per_style_total[style]
            for style in per_style_total
        }

        return accuracy, per_style_acc

    def _apply_params_to_classifier(self, params: ClassifierParams):
        """
        Apply parameters to the classifier.

        Now that StyleClassifier accepts params, we can just pass them in!
        """
        from neckenml.classifier import StyleClassifier

        # Recreate classifier with new params
        self.classifier = StyleClassifier(params=params)

    def optimize(
        self,
        target_accuracy: float = 0.90,
        max_iterations: int = 50,
        method: str = 'grid'
    ) -> ClassifierParams:
        """
        Optimize classifier parameters to reach target accuracy.

        Args:
            target_accuracy: Target overall accuracy (0.0-1.0)
            max_iterations: Maximum optimization iterations
            method: 'grid' for grid search, 'random' for random search,
                   'bayesian' for Bayesian optimization

        Returns:
            Optimized ClassifierParams
        """
        print(f"\n{'='*70}")
        print(f"AUTO-TUNING CLASSIFIER")
        print(f"{'='*70}")
        print(f"Target accuracy: {target_accuracy:.1%}")
        print(f"Max iterations: {max_iterations}")
        print(f"Method: {method}")
        print(f"{'='*70}\n")

        if method == 'grid':
            return self._grid_search(target_accuracy, max_iterations)
        elif method == 'random':
            return self._random_search(target_accuracy, max_iterations)
        elif method == 'bayesian':
            return self._bayesian_optimization(target_accuracy, max_iterations)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _grid_search(self, target_accuracy: float, max_iterations: int) -> ClassifierParams:
        """Grid search over key parameters."""
        best_params = ClassifierParams()
        best_train_accuracy = 0.0
        best_test_accuracy = 0.0
        best_per_style = {}

        # Define parameter grid (focus on most impactful parameters)
        param_grid = {
            'hambo_score_threshold': np.linspace(0.20, 0.45, 6),
            'schottis_swing_threshold': np.linspace(1.20, 1.50, 7),
            'vals_ratio_tolerance': np.linspace(0.05, 0.12, 8),
            'polska_score_fallback': np.linspace(0.30, 0.50, 5),
            'engelska_swing_max': np.linspace(0.85, 0.95, 6),
        }

        # Start with baseline
        print("Evaluating baseline parameters...")
        train_acc, _ = self.evaluate_params(best_params, self.train_data)
        test_acc, baseline_per_style = self.evaluate_params(best_params, self.test_data)
        print(f"Baseline - Train: {train_acc:.1%}, Test: {test_acc:.1%}\n")

        best_train_accuracy = train_acc
        best_test_accuracy = test_acc
        best_per_style = baseline_per_style

        iteration = 0

        # Grid search over each parameter
        for param_name, values in param_grid.items():
            if iteration >= max_iterations:
                break

            print(f"\nOptimizing {param_name}...")

            for value in values:
                iteration += 1

                # Create new params with this value
                params = ClassifierParams(**asdict(best_params))
                setattr(params, param_name, float(value))

                # Evaluate on TRAIN set (for optimization)
                train_acc, _ = self.evaluate_params(params, self.train_data)

                # Also evaluate on TEST set (for monitoring generalization)
                test_acc, per_style = self.evaluate_params(params, self.test_data)

                print(f"  [{iteration:3d}] {param_name}={value:.3f} → Train: {train_acc:.1%}, Test: {test_acc:.1%}", end='')

                # Optimize based on train accuracy, but monitor test
                if train_acc > best_train_accuracy:
                    best_train_accuracy = train_acc
                    best_test_accuracy = test_acc
                    best_params = params
                    best_per_style = per_style
                    print(" ✓ NEW BEST")

                    if train_acc >= target_accuracy:
                        print(f"\n🎉 Target train accuracy {target_accuracy:.1%} reached!")
                        print(f"    Test accuracy: {test_acc:.1%}")
                        break
                else:
                    print()

            if best_train_accuracy >= target_accuracy:
                break

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Training accuracy: {best_train_accuracy:.1%}")
        print(f"Test accuracy:     {best_test_accuracy:.1%}")
        print(f"Generalization:    {best_test_accuracy - best_train_accuracy:+.1%}")
        print(f"Iterations: {iteration}")
        print(f"\nPer-style accuracy (on test set):")
        for style, acc in sorted(best_per_style.items()):
            print(f"  {style:15} {acc:.1%}")
        print(f"{'='*70}\n")

        return best_params

    def _random_search(self, target_accuracy: float, max_iterations: int) -> ClassifierParams:
        """Random search over parameter space."""
        best_params = ClassifierParams()
        best_accuracy, best_per_style = self.evaluate_params(best_params)

        print(f"Baseline accuracy: {best_accuracy:.1%}\n")

        for iteration in range(max_iterations):
            # Generate random parameters (±30% from current best)
            params = self._mutate_params(best_params, mutation_rate=0.3)

            accuracy, per_style = self.evaluate_params(params)

            print(f"[{iteration+1:3d}/{max_iterations}] Accuracy: {accuracy:.1%}", end='')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                best_per_style = per_style
                print(" ✓ NEW BEST")

                if accuracy >= target_accuracy:
                    print(f"\n🎉 Target accuracy {target_accuracy:.1%} reached!")
                    break
            else:
                print()

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best accuracy: {best_accuracy:.1%}")
        print(f"\nPer-style accuracy:")
        for style, acc in sorted(best_per_style.items()):
            print(f"  {style:15} {acc:.1%}")
        print(f"{'='*70}\n")

        return best_params

    def _mutate_params(self, params: ClassifierParams, mutation_rate: float = 0.2) -> ClassifierParams:
        """Create mutated version of parameters."""
        new_params = ClassifierParams(**asdict(params))

        # Mutate each float parameter
        for field_name, field_value in asdict(params).items():
            if isinstance(field_value, float):
                # Random perturbation ±mutation_rate
                delta = field_value * mutation_rate * (2 * np.random.random() - 1)
                new_value = field_value + delta

                # Clamp to reasonable ranges
                if 'threshold' in field_name or 'score' in field_name:
                    new_value = np.clip(new_value, 0.1, 0.9)
                elif 'bpm' in field_name:
                    new_value = np.clip(new_value, 60, 200)
                elif 'swing' in field_name:
                    new_value = np.clip(new_value, 0.5, 2.5)
                elif 'tolerance' in field_name:
                    new_value = np.clip(new_value, 0.01, 0.20)

                setattr(new_params, field_name, float(new_value))
            elif isinstance(field_value, int) and field_name == 'polska_rescue_signals_required':
                # Mutate integer parameter
                new_value = np.clip(field_value + np.random.choice([-1, 0, 1]), 2, 6)
                setattr(new_params, field_name, int(new_value))

        return new_params

    def _bayesian_optimization(self, target_accuracy: float, max_iterations: int) -> ClassifierParams:
        """Bayesian optimization (requires scikit-optimize)."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
        except ImportError:
            print("⚠️  scikit-optimize not installed. Falling back to random search.")
            print("Install with: pip install scikit-optimize")
            return self._random_search(target_accuracy, max_iterations)

        # Define search space
        space = [
            Real(0.20, 0.45, name='hambo_score_threshold'),
            Real(1.20, 1.50, name='schottis_swing_threshold'),
            Real(0.05, 0.12, name='vals_ratio_tolerance'),
            Real(0.30, 0.50, name='polska_score_fallback'),
            Real(0.85, 0.95, name='engelska_swing_max'),
            Real(0.50, 0.60, name='polska_rescue_ternary_min'),
            Integer(3, 5, name='polska_rescue_signals_required'),
        ]

        # Objective function (minimize negative accuracy)
        def objective(param_values):
            params = ClassifierParams()
            params.hambo_score_threshold = param_values[0]
            params.schottis_swing_threshold = param_values[1]
            params.vals_ratio_tolerance = param_values[2]
            params.polska_score_fallback = param_values[3]
            params.engelska_swing_max = param_values[4]
            params.polska_rescue_ternary_min = param_values[5]
            params.polska_rescue_signals_required = param_values[6]

            accuracy, _ = self.evaluate_params(params)
            print(f"  Accuracy: {accuracy:.1%}")
            return -accuracy  # Minimize negative accuracy

        print("Running Bayesian optimization...")
        result = gp_minimize(
            objective,
            space,
            n_calls=max_iterations,
            random_state=42,
            verbose=True
        )

        # Extract best parameters
        best_params = ClassifierParams()
        best_params.hambo_score_threshold = result.x[0]
        best_params.schottis_swing_threshold = result.x[1]
        best_params.vals_ratio_tolerance = result.x[2]
        best_params.polska_score_fallback = result.x[3]
        best_params.engelska_swing_max = result.x[4]
        best_params.polska_rescue_ternary_min = result.x[5]
        best_params.polska_rescue_signals_required = result.x[6]

        best_accuracy, best_per_style = self.evaluate_params(best_params)

        print(f"\n{'='*70}")
        print(f"BAYESIAN OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best accuracy: {best_accuracy:.1%}")
        print(f"\nPer-style accuracy:")
        for style, acc in sorted(best_per_style.items()):
            print(f"  {style:15} {acc:.1%}")
        print(f"{'='*70}\n")

        return best_params

    def save_optimized_params(self, params: ClassifierParams, output_path: str = "optimized_params.json"):
        """Save optimized parameters to file."""
        with open(output_path, 'w') as f:
            json.dump(asdict(params), f, indent=2)
        print(f"✓ Optimized parameters saved to: {output_path}")

    def generate_code_changes(self, params: ClassifierParams, output_path: str = "OPTIMIZED_CHANGES.md"):
        """Generate markdown file with code changes needed."""
        changes = f"""# Optimized Classifier Parameters

## Generated Code Changes

Apply these changes to `neckenml/classifier/style_classifier.py`:

### 1. Hambo Detection (line ~174)
```python
if hambo_score > {params.hambo_score_threshold:.2f} and score_diff < -0.10:
```

### 2. Hambo Ratio Detection (line ~184)
```python
if ratios[0] > {params.hambo_ratio_threshold:.2f}:
```

### 3. Schottis Threshold (line ~246)
```python
if swing > {params.schottis_swing_threshold:.2f}:
```

### 4. Vals Detection (line ~191)
```python
if abs(ratios[0] - 0.33) < {params.vals_ratio_tolerance:.2f} and abs(ratios[1] - 0.33) < {params.vals_ratio_tolerance:.2f}:
    if polska_score < {params.vals_polska_score_max:.2f}:
```

### 5. Polska Fallback (line ~215)
```python
if polska_score > {params.polska_score_fallback:.2f}:
```

### 6. Polska Rescue Thresholds (line ~269, ~282, ~319)
```python
if ternary_conf < {params.polska_rescue_ternary_min:.2f}:
    return False

min_polska_score = 0.35 if ternary_conf >= 0.60 else {params.polska_rescue_polska_score_min:.2f}

return signals >= {params.polska_rescue_signals_required}
```

### 7. Mazurka Detection (line ~204)
```python
if swing > {params.mazurka_swing_high:.2f} or (swing > {params.mazurka_swing_medium:.2f} and ratios[1] > ratios[0]):
```

### 8. Menuett Detection (line ~238)
```python
if {params.menuett_swing_min:.2f} <= swing <= {params.menuett_swing_max:.2f}:
    if not bpm or bpm < {params.menuett_bpm_max:.0f}:
```

### 9. Engelska/Polka Detection (line ~271, ~276)
```python
if {params.engelska_swing_min:.2f} <= swing <= {params.engelska_swing_max:.2f}:
    return heuristic_result("Engelska", ...)
elif swing < {params.polka_swing_max:.2f}:
    return heuristic_result("Polka", ...)
```

### 10. Snoa Detection (line ~253, ~263)
```python
if bpm and {params.snoa_bpm_min:.0f} < bpm < {params.snoa_bpm_max:.0f}:
    if swing < {params.snoa_swing_max:.2f}:
        ...
elif not bpm and {params.snoa_swing_fallback_min:.2f} <= swing <= {params.snoa_swing_fallback_max:.2f}:
```

## Full Parameter Values

```json
{json.dumps(asdict(params), indent=2)}
```
"""

        with open(output_path, 'w') as f:
            f.write(changes)

        print(f"✓ Code changes saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Auto-tune classifier thresholds')
    parser.add_argument('--target', type=float, default=0.90,
                       help='Target accuracy (default: 0.90)')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum optimization iterations (default: 50)')
    parser.add_argument('--method', choices=['grid', 'random', 'bayesian'], default='grid',
                       help='Optimization method (default: grid)')
    parser.add_argument('--eval-data', default='test_data/evaluation_results.json',
                       help='Path to evaluation results')
    parser.add_argument('--output', default='optimized_params.json',
                       help='Output file for optimized parameters')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')

    args = parser.parse_args()

    try:
        # Create auto-tuner
        tuner = AutoTuningClassifier(
            args.eval_data,
            test_split=args.test_split,
            random_seed=args.random_seed
        )

        # Optimize
        best_params = tuner.optimize(
            target_accuracy=args.target,
            max_iterations=args.max_iterations,
            method=args.method
        )

        # Save results
        tuner.save_optimized_params(best_params, args.output)
        tuner.generate_code_changes(best_params, "OPTIMIZED_CHANGES.md")

        print("\n✓ Optimization complete!")
        print(f"\nNext steps:")
        print(f"1. Review OPTIMIZED_CHANGES.md")
        print(f"2. Apply the suggested changes to style_classifier.py")
        print(f"3. Re-run evaluation to verify improvements")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nMake sure you have run the evaluation first:")
        print(f"  python3 evaluate_classification.py")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
