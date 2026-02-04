# neckenml-analyzer

Audio analysis for Swedish folk music using Essentia, Madmom, and Librosa.

**AGPL-3.0 licensed**

## Installation

```bash
pip install neckenml-analyzer
```

This will also install `neckenml-core` as a dependency.

## What's Included

- **AudioAnalyzer**: Main class for analyzing audio files
- **Feature Extractors**: Rhythm, structure, swing, feel, vocal detection
- **Folk Authenticity**: Detector for traditional vs modern production

## Usage

```python
from neckenml.analyzer import AudioAnalyzer

with AudioAnalyzer() as analyzer:
    result = analyzer.analyze_file("path/to/audio.mp3")
    print(f"Detected style: {result['ml_suggested_style']}")
```

## License

AGPL-3.0 License - see [LICENSE](LICENSE) for details.

This package uses Essentia which is AGPL-licensed. If you need MIT-licensed components only, use `neckenml-core` instead.

## MIT Alternative

For classification without audio analysis (using pre-computed features):

```bash
pip install neckenml-core
```
