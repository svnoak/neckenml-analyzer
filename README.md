# NeckenML Analyzer

Swedish folk music audio analysis and dance style classification using machine learning.

## Overview

NeckenML Analyzer is a Python package that provides advanced audio analysis and automatic dance style classification for Swedish folk music. It uses a combination of signal processing, machine learning, and domain-specific heuristics to:

- **Analyze audio features**: BPM, meter (ternary/binary), swing ratio, vocal presence, articulation, bounciness, and more
- **Classify dance styles**: Polska, Hambo, Vals, Polka, Schottis, Snoa, and other Swedish folk dance types
- **Assess authenticity**: Distinguish traditional folk recordings from modern/electronic interpretations

## Features

- **Comprehensive Audio Analysis**
  - Tempo and beat detection using Madmom RNN (optimized for rubato in folk music)
  - Meter classification (3/4 ternary vs 2/4/4/4 binary)
  - MusiCNN embeddings for audio texture fingerprinting
  - Vocal vs instrumental detection
  - Swing ratio calculation
  - Articulation analysis (smooth/staccato/punchy)
  - Folk-specific features (Polska vs Hambo signatures)

- **Machine Learning Classification**
  - Pre-trained RandomForest classifier included
  - Hierarchical decision-making (metadata → ML → heuristics)
  - Confidence scores for each prediction
  - Support for model retraining with custom data

- **Extensible Architecture**
  - Abstract `AudioSource` interface for flexible audio acquisition
  - Built-in file-based source
  - Easy to implement custom sources (S3, HTTP, streaming, etc.)

## Installation

### 1. Install the package

```bash
pip install neckenml-analyzer
```

### 2. Set up PostgreSQL

neckenml Analyzer uses PostgreSQL with the pgvector extension for storing embeddings:

```bash
# Create database
createdb neckenml

# Enable pgvector extension
psql neckenml -c "CREATE EXTENSION vector;"
```

### 3. Download pre-trained models

The analyzer requires Essentia's MusiCNN models (not included due to licensing):

```bash
# Create models directory
mkdir -p ~/.neckenml/models

# Download MusiCNN embedding model
wget https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb \
  -O ~/.neckenml/models/msd-musicnn-1.pb

# Download voice/instrumental classifier
wget https://essentia.upf.edu/models/audio-event-recognition/voice_instrumental/voice_instrumental-musicnn-msd-1.pb \
  -O ~/.neckenml/models/voice_instrumental-musicnn-msd-1.pb
```

## Quick Start

```python
from neckenml import AudioAnalyzer, StyleClassifier
from neckenml.sources import FileAudioSource

# Set up audio source (file-based)
source = FileAudioSource(audio_dir="/path/to/your/audio/files")

# Initialize analyzer with audio source
analyzer = AudioAnalyzer(
    audio_source=source,
    model_dir="~/.neckenml/models"  # Optional, uses default if not specified
)

# Analyze an audio file (track_id should match filename without extension)
features = analyzer.analyze(track_id="my_track")

# The features dict contains:
# - bpm: Tempo in beats per minute
# - meter: 'ternary' or 'binary'
# - swing_ratio: 0.0-1.0 (0.5 = straight, 0.67 = triplet feel)
# - vocal_probability: 0.0-1.0 (vocal vs instrumental)
# - embedding: 217-dimensional feature vector
# - and many more...

# Classify dance style
classifier = StyleClassifier()
result = classifier.classify(features)

print(f"Detected style: {result['primary_style']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Secondary styles: {result['secondary_styles']}")

# Example output:
# Detected style: Polska
# Confidence: 85.0%
# Secondary styles: [('Slängpolska', 0.65)]
```

## Advanced Usage

### Artifact Persistence for Fast Re-analysis

Store expensive-to-compute artifacts once, then re-analyze instantly without touching audio files:

```python
from neckenml import AudioAnalyzer, compute_derived_features

# Initial analysis with artifact storage
result = analyzer.analyze_file(
    file_path="/path/to/track.mp3",
    return_artifacts=True
)

features = result["features"]          # Derived features
artifacts = result["raw_artifacts"]     # Raw data to store

# Store artifacts in database (AnalysisSource.raw_data JSONB column)
db.store(artifacts)

# Later: Fast re-analysis from stored artifacts (300x faster!)
new_features = compute_derived_features(artifacts)

# Re-classify with updated model (no audio needed!)
new_features = compute_derived_features(artifacts, new_classifier=my_model)
```

**Performance:** Re-classify 1000 tracks in ~2 minutes instead of 8+ hours!

See [Artifact Persistence Documentation](docs/artifact_persistence.md) for details.

### Custom Audio Source

Implement the `AudioSource` interface for custom audio acquisition:

```python
from neckenml.sources import AudioSource
import os

class CloudStorageAudioSource(AudioSource):
    """Fetch audio from cloud object storage"""

    def __init__(self, bucket_name, storage_client):
        self.client = storage_client
        self.bucket = bucket_name

    def fetch_audio(self, track_id: str) -> str:
        """Download audio file from cloud storage and return local path"""
        local_path = f"/tmp/{track_id}.mp3"
        self.client.download_file(
            bucket=self.bucket,
            key=f"audio/{track_id}.mp3",
            destination=local_path
        )
        return local_path

    def cleanup(self, file_path: str) -> None:
        """Clean up temporary file"""
        if os.path.exists(file_path):
            os.remove(file_path)

# Use custom source
source = CloudStorageAudioSource(bucket_name="my-music-bucket", storage_client=my_client)
analyzer = AudioAnalyzer(audio_source=source)
```

### Retraining the Classifier

Train a custom model with your own labeled data:

```python
from neckenml.training import TrainingService
import numpy as np

# Prepare training data
embeddings = np.array([...])  # Nx217 feature vectors from analyzer
labels = ["Polska", "Hambo", "Polska", ...]  # Dance style labels

# Train new model
trainer = TrainingService(model_path="./my_custom_model.pkl")
trainer.train_from_data(embeddings, labels)

# The classifier will automatically use the new model
```

## Supported Dance Styles

**Ternary (3/4 meter):**
- Polska
- Slängpolska
- Hambo
- Vals (Waltz)
- Springlek
- Mazurka

**Binary (2/4, 4/4 meter):**
- Polka
- Schottis
- Snoa
- Gånglåt
- Engelska
- Marsch

## Documentation

- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [Quick Start](docs/quickstart.md) - Getting started examples
- [Extending](docs/extending.md) - Custom AudioSource implementations and model training

## Architecture

NeckenML Analyzer uses a multi-stage pipeline:

1. **Audio Acquisition**: Flexible `AudioSource` interface
2. **Feature Extraction**: Madmom RNN for beat/rhythm analysis + Librosa for onsets
3. **Embedding Generation**: MusiCNN for 217-dim audio fingerprints
4. **Folk Features**: Domain-specific rhythm and meter analysis
5. **Classification**: Hierarchical decision tree (metadata → ML → heuristics)
6. **Artifact Persistence**: Store raw analysis outputs for instant re-classification

## Requirements

- Python 3.9+
- PostgreSQL with pgvector extension
- Essentia pre-trained models (see installation instructions)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- How to report bugs and suggest enhancements
- Development setup and coding standards
- Testing requirements and guidelines
- Pull request process

Whether you're fixing a bug, adding a feature, or improving documentation, your contributions help make Swedish folk music more accessible through technology.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use NeckenML Analyzer in your research, please cite:

```bibtex
@software{neckenml_analyzer,
  title = {NeckenML Analyzer: Swedish Folk Music Analysis and Classification},
  author = {NeckenML Contributors},
  year = {2025},
  url = {https://github.com/svnoak/neckenml-analyzer}
}
```

## Acknowledgments

- Built with [Essentia](https://essentia.upf.edu/) audio analysis library
- MusiCNN models by Jordi Pons et al.
- Powered by the Swedish folk music community
