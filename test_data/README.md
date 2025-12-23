# Dance Style Classification Testing Framework

This directory contains the complete testing and evaluation framework for improving dance style classification accuracy in neckenml-analyzer.

## Directory Structure

```
test_data/
├── README.md                          # This file
├── test_tracks.yaml                   # Test dataset definition
├── known_issues.yaml                  # Issue tracking and threshold change log
├── evaluation_results.json            # Latest evaluation results (generated)
├── baseline_results.json              # Baseline metrics (generated)
├── confusion_matrix_*.png             # Visualizations (generated)
└── audio/                             # Test audio files
    ├── polska/
    ├── polka/
    ├── hambo/
    ├── vals/
    ├── schottis/
    ├── snoa/
    ├── engelska/
    ├── slangpolska/
    └── ganglat/
```

## Quick Start

### 1. Curate Your Test Dataset

Add 4-5 audio files for each dance style to the appropriate `audio/` subfolder, then update [test_tracks.yaml](test_tracks.yaml) with the track information:

```yaml
tracks:
  - id: polska_001
    file_path: test_data/audio/polska/my_track.mp3
    true_style: Polska
    substyle: null
    difficulty: easy  # easy | medium | hard
    expected_bpm: ~110
    expected_meter: ternary
    notes: "Textbook Polska with clear beat 2 elongation"
    characteristics:
      - clear_asymmetry
      - rubato_timing
```

**Priority styles to test:**
- **Polska** (5+ tracks) - Include edge cases that might be misdetected as binary
- **Polka** (5+ tracks) - Include borderline tempo cases
- **Hambo** (4-5 tracks) - Include tracks with varying asymmetry
- **Vals** (4-5 tracks) - Include slower examples
- **Schottis** (3-4 tracks) - Varying swing levels
- **Snoa** (3-4 tracks) - Tempo boundary cases

### 2. Run Baseline Evaluation

```bash
cd /path/to/neckenml-analyzer

# Run full evaluation
python evaluate_classification.py --verbose

# Filter by difficulty
python evaluate_classification.py --difficulty hard

# Filter by style
python evaluate_classification.py --style Polska

# Generate confusion matrix plots
python evaluate_classification.py --plot
```

This will:
- Analyze all test tracks
- Generate accuracy metrics per style
- Identify critical misclassifications
- Save results to `evaluation_results.json`

### 3. Visualize Results

Generate comprehensive confusion matrix visualizations:

```bash
# Full visualization suite (4 plots)
python visualize_confusion_matrix.py

# Single normalized confusion matrix
python visualize_confusion_matrix.py --simple --normalize
```

Output:
- `confusion_matrix_counts.png` - Raw counts
- `confusion_matrix_normalized.png` - Percentages (recall)
- `confusion_matrix_errors_only.png` - Misclassifications highlighted
- `per_class_accuracy.png` - Bar chart of accuracy per style

### 4. Interactive Threshold Tuning

Open the Jupyter notebook for interactive analysis:

```bash
jupyter notebook threshold_tuning.ipynb
```

The notebook provides:
- Feature distribution analysis across styles
- Error pattern identification
- Interactive threshold experimentation
- Before/after comparison visualizations
- Change documentation templates

### 5. Iterate and Improve

1. **Identify errors** from evaluation report
2. **Analyze patterns** in the notebook (which features separate the confused styles?)
3. **Adjust thresholds** in neckenml classifier code
4. **Re-evaluate** to measure impact
5. **Document changes** in `known_issues.yaml`
6. **Repeat** until accuracy targets are met

## Key Files Explained

### test_tracks.yaml

Defines your ground truth dataset. Each track entry includes:
- `true_style` - Correct dance style (ground truth)
- `difficulty` - How challenging this case is (easy/medium/hard)
- `expected_bpm` / `expected_meter` - Expected analysis results
- `notes` - Why this track is in the test set
- `characteristics` - Tags for grouping similar cases

**Best practices:**
- Include at least 3-4 "easy" tracks per style (textbook examples)
- Include 1-2 "hard" tracks per style (edge cases, borderline)
- Focus on critical confusion pairs (Polska/Polka, Hambo/Polska)
- Document WHY each track is included

### known_issues.yaml

Tracks classification problems and improvement efforts:
- **Active issues** - Current classification failures
- **Threshold change log** - Every threshold adjustment with before/after metrics
- **Resolved issues** - Fixed problems for reference
- **Observations** - Insights from analysis
- **Baseline metrics** - Starting performance
- **Improvement targets** - Goals for accuracy

**Workflow:**
1. Document each error pattern as an issue
2. Log every threshold change with justification
3. Track metrics before/after each change
4. Close issues when resolved
5. Watch for regressions

### evaluate_classification.py

Automated evaluation script that:
- Loads test tracks from YAML
- Runs neckenml analysis on each track
- Compares predictions vs ground truth
- Categorizes errors by severity (critical/high/medium/low)
- Generates detailed metrics and reports

**Error severity levels:**
- **Critical** - Wrong meter (ternary ↔ binary) - would confuse dancers
- **High** - Major style confusion (Hambo → Polska)
- **Medium** - Moderate confusion (Vals → Polska)
- **Low** - Acceptable confusion (Engelska → Polka - rhythmically similar)

### visualize_confusion_matrix.py

Creates publication-quality visualizations:
- Standard confusion matrix with counts
- Normalized confusion matrix (shows recall per class)
- Error-focused view (correct predictions hidden)
- Per-class accuracy bar chart with color coding

### threshold_tuning.ipynb

Interactive Jupyter notebook for:
- **Feature analysis** - Distribution plots, box plots, scatter plots
- **Error analysis** - Deep dive into misclassifications
- **Threshold experiments** - Test changes before committing to code
- **Comparison tools** - Before/after metrics side-by-side
- **Documentation** - Templates for change logs

## Critical Thresholds to Tune

Based on the codebase analysis, focus on these key thresholds:

### Polska Detection
```python
# In neckenml rhythm_extractor.py
POLSKA_SCORE_MIN = 0.45        # Strong Polska signal
POLSKA_SCORE_WEAK = 0.25       # Weak Polska signal
RATIO_ELONGATION = 0.34        # Beat 2/3 elongation threshold
TIMING_VARIANCE_MIN = 0.003    # Rubato indicator
```

### Polska Rescue Logic (Binary → Ternary)
```python
# In neckenml style_classifier.py
RESCUE_TERNARY_MIN = 0.45      # Minimum ternary confidence to attempt rescue
RESCUE_TERNARY_STRONG = 0.65   # Adds +2 signals
RESCUE_TERNARY_MODERATE = 0.55 # Adds +1 signal
RESCUE_SIGNALS_NEEDED = 3      # Minimum signals to rescue
POLSKA_BPM_RANGE = (95, 115)   # Typical Polska tempo
```

### Hambo Detection
```python
HAMBO_SCORE_MIN = 0.45
HAMBO_LONG_FIRST_BEAT = 0.40       # r1 threshold
HAMBO_POLSKA_SEPARATION = 0.10     # Must exceed polska_score by this much
```

### Binary Styles
```python
SCHOTTIS_SWING_MIN = 1.25      # High swing = Schottis
SNOA_TEMPO_RANGE = (80, 115)   # Walking tempo
POLKA_TEMPO_MIN = 115          # Fast tempo
```

## Interpretation Guide

### Reading Confusion Matrices

Rows = True style, Columns = Predicted style

```
              Predicted
           Polka  Polska  Hambo
True Polka    4      0      0     ← Perfect Polka classification
     Polska   1      3      0     ← 1 Polska misclassified as Polka (CRITICAL)
     Hambo    0      1      3     ← 1 Hambo misclassified as Polska
```

**What to look for:**
- Diagonal = correct predictions (should be high)
- Off-diagonal = errors (should be minimal)
- Critical errors = different meter (ternary ↔ binary)

### Accuracy Targets

**Overall:** >90%

**Per-style minimums:**
- Polska: >85% (critical style, challenging to detect)
- Polka: >90% (common style, should be reliable)
- Hambo: >85% (distinctive but can overlap with Polska)
- Vals: >90% (very distinctive tempo/feel)
- Schottis: >80% (swing-based, can overlap with Polka)
- Snoa: >75% (tempo-based, fuzzy boundary with Polka)

**Acceptable confusions:**
- Engelska ↔ Polka (rhythmically identical)
- Snoa ↔ Polka (tempo boundary)
- Gånglåt ↔ Marsch (similar character)

**UNACCEPTABLE confusions:**
- Polska → Polka (different meter!)
- Hambo → Polka (different meter!)
- Any ternary → binary (or vice versa)

## Workflow Example

### Day 1: Setup and Baseline

```bash
# 1. Add test tracks to audio/ folders
cp ~/Music/confirmed_polska/*.mp3 test_data/audio/polska/

# 2. Update test_tracks.yaml with track info
nano test_data/test_tracks.yaml

# 3. Run baseline evaluation
python evaluate_classification.py --verbose > baseline_report.txt

# 4. Generate visualizations
python visualize_confusion_matrix.py
```

### Day 2-3: Analysis and Tuning

```bash
# 1. Open notebook for analysis
jupyter notebook threshold_tuning.ipynb

# 2. Identify top issues (e.g., Polska → Polka errors)

# 3. Modify thresholds in neckenml code:
#    - Edit neckenml/classifier/style_classifier.py
#    - Reduce rescue_ternary_min from 0.45 to 0.42

# 4. Re-evaluate
python evaluate_classification.py --verbose

# 5. Compare metrics in notebook

# 6. Document change in known_issues.yaml
```

### Week 2: Expand Dataset

```bash
# 1. Add 5-10 more tracks per style
# 2. Re-run full evaluation
# 3. Validate that improvements hold on new data
```

## Tips for Success

1. **Start with clear examples** - Use textbook tracks for your first test set
2. **Add edge cases gradually** - Once basics work, add challenging tracks
3. **Focus on critical errors first** - Polska/Polka confusion is worse than Schottis/Polka
4. **Document everything** - Track every threshold change and its impact
5. **Watch for regressions** - Fixing Polska might break Polka - check both!
6. **Use multiple test rounds** - Don't tune on the same tracks you evaluate on
7. **Trust your ears** - If the classification feels wrong, it probably is

## Advanced: ML Model Retraining

Once you have good heuristic performance, consider retraining the RandomForest model:

```bash
# In your backend, if using the user feedback training system:
python -m app.services.training retrain --min-confirmations 2

# This will:
# - Query confirmed dance styles from database
# - Extract embeddings from analysis data
# - Train new RandomForest classifier
# - Save as custom_style_head.pkl
```

The ML model learns from user feedback, so more confirmed classifications = better model.

## Troubleshooting

**"No module named neckenml"**
- Make sure neckenml is installed: `pip install -e /path/to/neckenml`

**"Test data file not found"**
- Run from neckenml-analyzer root directory
- Or specify full path: `python evaluate_classification.py --test-data /full/path/to/test_tracks.yaml`

**"File not found" for audio tracks**
- Check file paths in test_tracks.yaml are correct
- Paths should be relative to neckenml-analyzer root

**Evaluation is slow**
- First run extracts features (slow), subsequent runs should be faster
- Use `--style Polska` to test only one style during development

## Questions?

Refer to the main analysis document for detailed information about:
- How the classification algorithm works
- What features are extracted
- Current threshold values in the code
- Decision path logic (metadata → AI → heuristics)
