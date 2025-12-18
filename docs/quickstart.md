# Quick Start Guide

## Basic Usage

### 1. Analyze a Single File

The simplest way to analyze an audio file:

```python
from neckenml import AudioAnalyzer, StyleClassifier

# Initialize analyzer
analyzer = AudioAnalyzer()

# Analyze a file directly (no AudioSource needed)
features = analyzer.analyze_file("/path/to/your/song.mp3")

# Classify the dance style
classifier = StyleClassifier()
result = classifier.classify(features)

print(f"Style: {result['primary_style']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"BPM: {features['tempo_bpm']}")
print(f"Meter: {features['meter']}")
```

### 2. Analyze Multiple Files

Use `FileAudioSource` for batch processing:

```python
from neckenml import AudioAnalyzer, StyleClassifier
from neckenml.sources import FileAudioSource

# Set up file-based source
source = FileAudioSource(audio_dir="/path/to/audio/folder")
analyzer = AudioAnalyzer(audio_source=source)
classifier = StyleClassifier()

# Analyze multiple tracks
track_ids = ["song1", "song2", "song3"]  # Filenames without extension

for track_id in track_ids:
    # Fetch and analyze
    features = analyzer.analyze(track_id)

    # Classify
    result = classifier.classify(features)

    print(f"{track_id}: {result['primary_style']} ({result['confidence']:.0%})")
```

## Understanding the Results

### Audio Features

The `analyze()` method returns a dict with:

```python
{
    # Rhythm features
    'tempo_bpm': 120.5,              # Beats per minute
    'meter': '3/4',                  # Time signature
    'swing_ratio': 0.55,             # 0.5=straight, 0.67=triplet
    'bpm_stability': 0.95,           # Tempo consistency

    # Timbre features
    'voice_probability': 0.2,        # 0-1 (vocal presence)
    'is_likely_instrumental': True,
    'loudness_lufs': -14.0,         # Loudness (LUFS)

    # Feel features
    'articulation': 0.6,             # Smooth/punchy/staccato
    'bounciness': 0.7,               # Rhythmic pulse

    # Folk-specific
    'polska_score': 0.85,            # Polska rhythm signature
    'hambo_score': 0.3,              # Hambo rhythm signature
    'ternary_confidence': 0.9,       # 3/4 vs 4/4 confidence

    # ML features
    'ml_suggested_style': 'Polska',
    'ml_confidence': 0.85,
    'embedding': [0.1, 0.2, ...],   # 217-dim feature vector

    # Structure
    'bars': [0.5, 1.0, 1.5, ...],   # Bar positions in seconds
    'sections': [8.0, 16.0, ...],   # Section boundaries
    'section_labels': ['A', 'B', ...],

    # Authenticity
    'folk_authenticity_score': 0.8,
    'requires_manual_review': False,
}
```

### Classification Results

The `classify()` method returns:

```python
{
    'primary_style': 'Polska',
    'confidence': 0.85,
    'source': 'ml_model',           # 'metadata', 'ml_model', or 'heuristic'
    'secondary_styles': [
        ('Slängpolska', 0.65),
        ('Hambo', 0.45)
    ],
    'tempo_category': 'moderate',    # 'slow', 'moderate', 'fast'
    'bpm_multiplier': 1.0,
    'effective_bpm': 120
}
```

## Storing Results in Database

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from neckenml.models.schema import Base, Track, TrackDanceStyle, AnalysisSource

# Set up database
engine = create_engine('postgresql://localhost/neckenml')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db = Session()

# Create track
track = Track(
    title="Example Polska",
    artist="Traditional",
    duration_ms=180000
)
db.add(track)
db.commit()

# Analyze
features = analyzer.analyze_file("/path/to/polska.mp3")
result = classifier.classify(features)

# Store analysis
analysis = AnalysisSource(
    track_id=track.id,
    source_type='hybrid_ml_v2',
    raw_data=features,
    confidence_score=result['confidence']
)
db.add(analysis)

# Store classification
style = TrackDanceStyle(
    track_id=track.id,
    dance_style=result['primary_style'],
    confidence=result['confidence'],
    embedding=features['embedding']
)
db.add(style)

db.commit()
print(f"Saved {track.title} as {style.dance_style}")
```

## Training a Custom Model

If you have labeled data, you can train your own classifier:

```python
from neckenml.training import TrainingService
import numpy as np

# Prepare training data
# (In reality, you'd get this from analyzing many files)
embeddings = []
labels = []

for file_path, label in training_data:
    features = analyzer.analyze_file(file_path)
    embeddings.append(features['embedding'])
    labels.append(label)  # 'Polska', 'Hambo', etc.

# Train
trainer = TrainingService()
success = trainer.train_from_data(
    embeddings=np.array(embeddings),
    labels=labels
)

if success:
    print("Model trained successfully!")

# The classifier will now use your custom model
result = classifier.classify(features)
```

## Next Steps

- [Extending Guide](extending.md) - Implement custom AudioSource, train models
