# neckenml

Swedish folk music audio analysis and dance style classification.

This is the full package that includes both core classification and audio analysis.

**AGPL-3.0 licensed** - using this package requires your application to be AGPL-licensed if distributed.

## Installation

```bash
pip install neckenml
```

## Quick Start

```python
from neckenml import AudioAnalyzer, StyleClassifier

# Analyze an audio file
with AudioAnalyzer() as analyzer:
    result = analyzer.analyze_file("path/to/track.mp3")
    print(f"Style: {result['ml_suggested_style']}")
    print(f"BPM: {result['tempo_bpm']}")

# Classify with the full classifier
classifier = StyleClassifier()
classifications = classifier.classify(track, result)
```

## Package Structure

This metapackage installs:

| Package | License | Description |
|---------|---------|-------------|
| `neckenml-core` | MIT | Database models, classifier, training |
| `neckenml-analyzer` | AGPL | Audio analysis with Essentia |

## MIT Alternative

If you only need classification (without audio analysis), install the MIT-licensed core:

```bash
pip install neckenml-core
```

```python
from neckenml.core import StyleClassifier, Track, TrainingService
```

## License

AGPL-3.0 License - see [LICENSE](LICENSE) for details.
