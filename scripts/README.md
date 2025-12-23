# neckenml-analyzer Scripts

Command-line tools for evaluating and optimizing the dance style classifier.

## Evaluation Tools

### evaluate_classification.py
Main evaluation script that tests classifier accuracy against ground truth data.

```bash
python scripts/evaluate_classification.py test_data/evaluation_results.json
```

Creates detailed evaluation reports including:
- Per-style accuracy metrics
- Confusion matrix
- Feature distribution analysis

### visualize_confusion_matrix.py
Generate visual confusion matrix from evaluation results.

```bash
python scripts/visualize_confusion_matrix.py evaluation_report.json
```

Outputs `confusion_matrix.png` showing misclassification patterns.

## Optimization Tools

### auto_tune_classifier.py
Automatically optimize classifier parameters using train/test split.

```bash
# Basic usage
python scripts/auto_tune_classifier.py test_data/evaluation_results.json

# With custom settings
python scripts/auto_tune_classifier.py test_data/evaluation_results.json \
    --target 0.85 \
    --max-iterations 50 \
    --method random \
    --test-split 0.2 \
    --random-seed 42
```

**Important**: Requires 100+ tracks for meaningful optimization. With smaller datasets, use manual parameter adjustments instead.

**Methods**:
- `grid`: Systematic grid search (slow but thorough)
- `random`: Random parameter sampling (faster)
- `bayesian`: Bayesian optimization (experimental)

### analyze_eval_features.py
Analyze feature distributions per dance style to inform manual tuning.

```bash
python scripts/analyze_eval_features.py test_data/evaluation_results.json
```

Outputs statistical analysis of features (BPM, swing ratio, scores) grouped by style.

### rerun_classification.py
Re-classify tracks with new parameters without re-analyzing audio.

```bash
python scripts/rerun_classification.py test_data/evaluation_results.json \
    --params config/classifier_params.json
```

Useful for quickly testing parameter changes.

## Parameter Files

Optimized parameters are saved in JSON format:

```json
{
  "version": 2,
  "hambo_score_threshold": 0.30,
  "schottis_swing_threshold": 1.35,
  "training_samples": 150,
  "accuracy_on_training": 0.87
}
```

Load them in your code:

```python
from neckenml.classifier import StyleClassifier
from neckenml.classifier.params import ClassifierParams

params = ClassifierParams.load('config/classifier_params.json')
classifier = StyleClassifier(params=params)
```

## Workflow

1. **Collect ground truth data** - Create evaluation dataset with known styles
2. **Baseline evaluation** - Run `evaluate_classification.py` to get current accuracy
3. **Analyze features** - Run `analyze_eval_features.py` to understand patterns
4. **Manual tuning** - Adjust parameters based on analysis (small datasets)
5. **Auto-optimization** - Use `auto_tune_classifier.py` once you have 100+ tracks
6. **Validation** - Test on held-out data using `--test-split`
7. **Deploy** - Load optimized params in production classifier
