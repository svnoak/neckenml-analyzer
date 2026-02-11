# NeckenML

Audio analysis and dance style classification for Swedish folk music.

## License

This repository is **dual-licensed** by package:

| Package | License | Use in proprietary software |
|---------|---------|-----------------------------|
| **neckenml-core** | [MIT](packages/neckenml-core/LICENSE) | Yes (classification only, no audio stack) |
| **neckenml-analyzer** | [AGPL-3.0](LICENSE) | No (copyleft; network use triggers source sharing) |
| **neckenml** (meta) | [AGPL-3.0](LICENSE) | No (pulls in analyzer) |

Use **neckenml-core** when you only need style classification from pre-computed features and want to keep your project under a non-copyleft license. Use **neckenml** or **neckenml-analyzer** when you need full audio analysis and can comply with AGPL.

## Quick Start

```bash
pip install neckenml
```

```python
from neckenml import AudioAnalyzer

with AudioAnalyzer() as analyzer:
    result = analyzer.analyze_file("track.mp3")

print(f"Style: {result['ml_suggested_style']}")
print(f"Tempo: {result['tempo_bpm']} BPM")
print(f"Confidence: {result['ml_confidence']:.0%}")
```

## Packages

| Package | License | Install | Use Case |
|---------|---------|---------|----------|
| `neckenml` | AGPL-3.0 | `pip install neckenml` | Full audio analysis |
| `neckenml-core` | MIT | `pip install neckenml-core` | Classification only (no audio) |
| `neckenml-analyzer` | AGPL-3.0 | `pip install neckenml-analyzer` | Audio analysis engine |

**Use `neckenml-core`** for proprietary projects that only need classification from pre-computed features.

**Use `neckenml`** when you need to analyze audio files (requires AGPL compliance).

## Requirements

For audio analysis, install system dependencies first:

```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1 ffmpeg

# macOS
brew install libsndfile ffmpeg
```

Download ML models:

```bash
mkdir -p ~/.neckenml/models
curl -L -o ~/.neckenml/models/msd-musicnn-1.pb \
  https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb
curl -L -o ~/.neckenml/models/voice_instrumental-musicnn-msd-1.pb \
  https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1.pb
```

## Supported Dance Styles

| Ternary (3/4) | Binary (2/4, 4/4) |
|---------------|-------------------|
| Polska | Polka |
| Slängpolska | Schottis |
| Hambo | Snoa |
| Vals | Gånglåt |
| Mazurka | Engelska |

## Core-Only Usage

For classification without audio analysis (MIT licensed):

```python
from neckenml.core import StyleClassifier, compute_derived_features

# Re-analyze from stored artifacts (no audio needed)
features = compute_derived_features(stored_raw_artifacts)

classifier = StyleClassifier()
results = classifier.classify(track, features)
```

## Development

```bash
git clone https://github.com/svnoak/neckenml-analyzer.git
cd neckenml-analyzer

pip install -e packages/neckenml-core
pip install -e packages/neckenml-analyzer
pip install pytest

pytest
```

## Links

- [Contributing Guide](CONTRIBUTING.md)
