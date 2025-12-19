# Artifact Persistence and Fast Re-analysis

## Overview

NeckenML Analyzer supports **artifact persistence** - storing expensive-to-compute intermediate results that enable fast re-analysis without re-processing audio files.

This dramatically improves iteration speed when:
- Updating classification models
- Tweaking feature engineering logic
- A/B testing scoring algorithms
- Bulk re-classifying your entire database

## The Problem

Traditional audio analysis requires re-running expensive operations every time:

```python
# Change classification logic
def updated_classification(features):
    # New algorithm here
    pass

# Have to re-analyze ALL tracks from scratch
for track in tracks:
    features = analyzer.analyze_file(track.path)  # 30 seconds per track
    classify(features)
```

**For 1000 tracks: 8+ hours** 😱

## The Solution

Store expensive artifacts once, re-compute derived features instantly:

```python
# Initial analysis (once)
result = analyzer.analyze_file(track.path, return_artifacts=True)
db.store(result["raw_artifacts"])  # Store in AnalysisSource.raw_data

# Fast re-analysis (as many times as needed)
features = compute_derived_features(raw_artifacts)  # 0.1 seconds
classify(features)
```

**For 1000 tracks: ~2 minutes** 🚀

**300x faster!**

## What Gets Stored

The `raw_artifacts` dictionary contains:

### 1. Madmom Beat Detection (Most Expensive)
```python
"madmom": {
    "beat_times": [0.5, 1.2, 1.9, ...],      # Beat positions
    "beat_info": [[0.5, 1], [1.2, 2], ...],  # Beat numbers
    "ternary_confidence": 0.85,              # Meter classification
    "bars": [0.5, 2.9, 5.3, ...]            # Bar positions
}
```

### 2. Neural Network Outputs
```python
"musicnn": {
    "avg_embedding": [200-dimensional vector],     # MusiCNN features
    "raw_embeddings": [[...], [...], ...]          # Time-series (optional)
},
"vocal": {
    "instrumental_score": 0.8,
    "vocal_score": 0.2,
    "predictions": [[...], [...]]  # Time-series (optional)
}
```

### 3. Audio Statistics
```python
"audio_stats": {
    "loudness_lufs": -14.0,
    "rms": 0.15,
    "zcr": 0.05,
    "onset_rate": 2.5,
    "duration_seconds": 180.0
}
```

### 4. Sub-beat Analysis
```python
"onsets": {
    "librosa_onset_times": [0.1, 0.6, 1.1, ...]  # For swing calculation
}
```

### 5. Dynamics
```python
"dynamics": {
    "beat_activations": [0.8, 0.6, 0.9, ...],  # Energy at each beat
    "envelope": [downsampled amplitude envelope],
    "intervals": [0.5, 0.6, 0.5, ...]          # Beat intervals
}
```

## What Gets Re-computed

These are **fast math operations** on stored vectors:

- ⚡ BPM, swing ratio, punchiness
- ⚡ Polska/Hambo signature scores
- ⚡ Articulation, bounciness
- ⚡ Full 217-dimensional feature vector
- ⚡ ML classification with any model
- ⚡ Folk authenticity score

## Usage

### Basic Usage

```python
from neckenml import AudioAnalyzer, compute_derived_features

# 1. Initial analysis with artifacts
analyzer = AudioAnalyzer(audio_source=source)
result = analyzer.analyze_file(
    file_path="/path/to/track.mp3",
    return_artifacts=True
)

features = result["features"]
artifacts = result["raw_artifacts"]

# 2. Store artifacts in database
analysis_source = AnalysisSource(
    track_id=track.id,
    source_type="neckenml_analyzer",
    raw_data=artifacts  # JSONB column
)
session.add(analysis_source)
session.commit()

# 3. Later: Fast re-analysis from stored artifacts
raw_artifacts = analysis_source.raw_data
new_features = compute_derived_features(raw_artifacts)
```

### With Custom Classifier

```python
from neckenml.classifier.style_head import ClassificationHead

# Train or load a new model
new_classifier = ClassificationHead(model_path="./my_new_model.pkl")

# Re-classify with new model (no audio needed!)
features = compute_derived_features(
    raw_artifacts,
    new_classifier=new_classifier
)
```

### Bulk Re-classification

```python
from sqlalchemy.orm import Session
from neckenml.models import AnalysisSource
from neckenml import compute_derived_features

def bulk_reclassify(session: Session, new_classifier=None):
    """Re-classify all tracks without touching audio files."""

    # Query all stored artifacts
    analysis_sources = session.query(AnalysisSource).all()

    for source in analysis_sources:
        # Fast re-analysis (0.1 seconds per track)
        new_features = compute_derived_features(
            source.raw_data,
            new_classifier=new_classifier
        )

        # Store updated classification
        style = TrackDanceStyle(
            track_id=source.track_id,
            dance_style=new_features['ml_suggested_style'],
            confidence=new_features['ml_confidence'],
            embedding=new_features['embedding']
        )
        session.add(style)

    session.commit()
    print(f"Re-classified {len(analysis_sources)} tracks!")

# Re-classify 1000 tracks in ~2 minutes instead of 8 hours
bulk_reclassify(session, new_classifier=my_improved_model)
```

## Storage Size

Typical storage per track:

- **Minimal** (embeddings + stats only): ~2 KB
- **Standard** (+ beat data): ~5 KB
- **Full** (+ time-series data): ~20 KB

For 10,000 tracks:
- Minimal: 20 MB
- Standard: 50 MB
- Full: 200 MB

This is negligible compared to the **hours of computation time** saved.

## Migration Strategy

If you have existing tracks without artifacts:

```python
# Backfill artifacts for existing tracks
def backfill_artifacts(session: Session):
    analyzer = AudioAnalyzer(audio_source=source)

    # Find tracks without artifacts
    tracks_without_artifacts = session.query(Track).outerjoin(
        AnalysisSource
    ).filter(
        AnalysisSource.id.is_(None)
    ).all()

    for track in tracks_without_artifacts:
        print(f"Analyzing {track.title}...")

        # Analyze with artifacts
        result = analyzer.analyze_file(
            track.file_path,
            return_artifacts=True
        )

        # Store artifacts
        analysis_source = AnalysisSource(
            track_id=track.id,
            source_type="neckenml_analyzer",
            raw_data=result["raw_artifacts"]
        )
        session.add(analysis_source)

    session.commit()
```

## Best Practices

### 1. Always Store Artifacts for New Tracks

```python
# Good ✓
result = analyzer.analyze_file(path, return_artifacts=True)
store_in_db(result["raw_artifacts"])

# Bad ✗ (will have to re-analyze later)
result = analyzer.analyze_file(path)
```

### 2. Version Your Artifact Schema

The artifacts include a version field for future compatibility:

```python
artifacts["version"]  # "1.0.0"
```

If you update the artifact structure, increment the version and handle migrations.

### 3. Keep Time-Series Data Optional

Large arrays (raw embeddings, predictions) are marked optional. For most use cases, you only need the aggregated values.

### 4. Re-analyze When Audio Changes

If the source audio is modified (re-mastered, re-encoded), re-run the full analysis:

```python
if track.audio_updated_at > analysis_source.analyzed_at:
    # Re-analyze from scratch
    result = analyzer.analyze_file(path, return_artifacts=True)
    analysis_source.raw_data = result["raw_artifacts"]
```

## Performance Benchmarks

Measured on a 3-minute folk tune:

| Operation | Time | Notes |
|-----------|------|-------|
| Full audio analysis | 25-35s | Madmom + MusiCNN + extractors |
| Re-analysis from artifacts | 0.05-0.15s | Just math on stored vectors |
| **Speedup** | **~300x** | |

For 1000 tracks:
- Full analysis: **8.3 hours**
- Re-analysis: **1.7 minutes**

## Example Workflows

### Workflow 1: Model Development

```python
# Day 1: Analyze all tracks once
for track in tracks:
    result = analyzer.analyze_file(track.path, return_artifacts=True)
    db.store(result["raw_artifacts"])

# Days 2-30: Iterate on classification model
for experiment in range(100):
    model = train_new_model(experiment_params)

    # Re-classify all tracks in minutes
    for artifacts in db.get_all_artifacts():
        features = compute_derived_features(artifacts, new_classifier=model)
        evaluate(features)
```

### Workflow 2: A/B Testing Algorithms

```python
# Test two different scoring algorithms
def algorithm_a(artifacts):
    return compute_derived_features(artifacts)

def algorithm_b(artifacts):
    # Custom feature engineering
    features = compute_derived_features(artifacts)
    features['polska_score'] *= 1.2  # Boost polska
    return features

# Compare on entire dataset (fast!)
for artifacts in db.get_all_artifacts():
    result_a = algorithm_a(artifacts)
    result_b = algorithm_b(artifacts)
    compare(result_a, result_b)
```

## See Also

- [Database Schema Documentation](../neckenml/models/schema.py)
- [Example: Artifact Persistence](../examples/artifact_persistence_example.py)
- [API Reference: compute_derived_features](../neckenml/analyzer/reanalysis.py)
