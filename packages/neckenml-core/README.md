# neckenml-core

Core components for Swedish folk music classification. **MIT licensed** - safe for use in proprietary applications.

## Installation

```bash
pip install neckenml-core
```

## What's Included

- **Database Models**: `Track`, `AnalysisSource`, `TrackDanceStyle` (SQLAlchemy)
- **Classifier**: `StyleClassifier`, `ClassificationHead`, `ClassifierParams`
- **Audio Sources**: `AudioSource` (interface), `FileAudioSource`
- **Training**: `TrainingService` for model training

## Usage

```python
from neckenml.core import StyleClassifier, ClassificationHead, Track

# Use pre-computed feature vectors for classification
classifier = StyleClassifier()
results = classifier.classify(track, analysis_data)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Full Package

For audio analysis capabilities (requires AGPL-licensed Essentia), install the full package:

```bash
pip install neckenml  # or pip install neckenml-analyzer
```
