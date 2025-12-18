# Contributing to Neckenml Analyzer

Thank you for your interest in contributing to Neckenml Analyzer! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

This project is dedicated to providing a welcoming and inclusive experience for everyone. We expect all contributors to:

- Be respectful and considerate in communication
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

**When submitting a bug report, include:**
- Clear, descriptive title
- Steps to reproduce the behavior
- Expected vs actual behavior
- Audio file format and characteristics (if applicable)
- Python version and operating system
- Full error traceback

**Example:**
```
Title: Incorrect BPM detection for 6/8 meter

Description:
When analyzing a 6/8 meter polska, the BPM is detected as double the actual tempo.

Steps to reproduce:
1. Analyze a 6/8 meter audio file
2. Check the returned 'tempo_bpm' value

Expected: ~120 BPM
Actual: ~240 BPM

Environment:
- Python 3.10
- Ubuntu 22.04
- neckenml-analyzer 0.1.0
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- Clear use case description
- Why this enhancement would be useful
- Possible implementation approach (if you have ideas)

**Areas particularly welcome for enhancement:**
- New folk dance style detection
- Improved meter classification
- Additional audio features
- Performance optimizations
- Documentation improvements

### Code Contributions

We welcome code contributions! Here are some areas where help is needed:

**High Priority:**
- Improving classification accuracy for rare dance styles
- Better handling of live recordings vs studio recordings
- Support for additional audio formats
- Performance optimizations for batch processing

**Medium Priority:**
- Additional unit tests and test fixtures
- Documentation improvements and examples
- Custom AudioSource implementations (examples)
- Integration examples with popular frameworks

**Good First Issues:**
- Documentation typos and clarifications
- Code comments and docstrings
- Example scripts
- Test fixtures with CC-licensed folk music

## Development Setup

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 12+ with pgvector extension
- Git

### Setting Up Your Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone <your-fork-url>
   cd neckenml-analyzer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev,audio]"
   ```

4. **Set up PostgreSQL:**
   ```bash
   createdb neckenml_test
   psql neckenml_test -c "CREATE EXTENSION vector;"
   ```

5. **Download required models:**
   ```bash
   mkdir -p ~/.neckenml/models

   # MusiCNN embedding model
   wget https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb \
     -O ~/.neckenml/models/msd-musicnn-1.pb

   # Voice/Instrumental classifier
   wget https://essentia.upf.edu/models/audio-event-recognition/voice_instrumental/voice_instrumental-musicnn-msd-1.pb \
     -O ~/.neckenml/models/voice_instrumental-musicnn-msd-1.pb
   ```

6. **Verify installation:**
   ```bash
   python -c "from neckenml import AudioAnalyzer; print('✅ Setup complete!')"
   ```

### Project Structure

```
neckenml-analyzer/
├── neckenml/
│   ├── analyzer/          # Core audio analysis
│   │   ├── audio_analyzer.py
│   │   └── extractors/    # Feature extractors
│   ├── classifier/        # Style classification
│   ├── sources/          # Audio source implementations
│   ├── training/         # Model training
│   └── models/           # Database models
├── tests/                # Unit tests
│   ├── test_analyzer.py
│   ├── test_classifier.py
│   └── fixtures/         # Test audio files
├── docs/                 # Documentation
├── examples/             # Example scripts
└── pyproject.toml        # Package configuration
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length:** 100 characters maximum
- **Indentation:** 4 spaces (no tabs)
- **Imports:** Group in order: standard library, third-party, local
- **Naming conventions:**
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_CASE`
  - Private methods: `_leading_underscore`

### Code Formatting

We use **Black** for code formatting:

```bash
# Format your code
black neckenml/ tests/

# Check formatting
black --check neckenml/ tests/
```

### Type Hints

Use type hints for function signatures:

```python
def analyze_file(self, file_path: str, metadata_context: str = "") -> dict[str, Any]:
    """Analyze an audio file and return features."""
    pass
```

### Documentation

All public functions, classes, and modules should have docstrings:

```python
def classify(self, features: dict) -> dict:
    """
    Classify dance style from audio features.

    Args:
        features: Dictionary of audio features from AudioAnalyzer

    Returns:
        Dictionary containing:
            - primary_style (str): Detected dance style
            - confidence (float): Confidence score 0.0-1.0
            - secondary_styles (list): Alternative classifications
            - source (str): 'metadata', 'ml_model', or 'heuristic'

    Example:
        >>> classifier = StyleClassifier()
        >>> features = analyzer.analyze_file("polska.mp3")
        >>> result = classifier.classify(features)
        >>> print(result['primary_style'])
        'Polska'
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neckenml --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py

# Run specific test
pytest tests/test_analyzer.py::test_tempo_detection
```

### Writing Tests

**Test file naming:** `test_<module>.py`

**Test function naming:** `test_<functionality>`

**Example test:**
```python
import pytest
from neckenml import AudioAnalyzer
from neckenml.sources import FileAudioSource

def test_tempo_detection_accuracy():
    """Test BPM detection on known tempo files."""
    source = FileAudioSource(audio_dir="tests/fixtures")
    analyzer = AudioAnalyzer(audio_source=source)

    # Test file has known BPM of 120
    features = analyzer.analyze("known_120bpm")

    assert 115 <= features['tempo_bpm'] <= 125, \
        f"Expected ~120 BPM, got {features['tempo_bpm']}"

def test_meter_classification():
    """Test ternary vs binary meter detection."""
    analyzer = AudioAnalyzer()

    # Test 3/4 meter file
    features = analyzer.analyze_file("tests/fixtures/polska_34.mp3")
    assert features['meter'] == 'ternary'

    # Test 2/4 meter file
    features = analyzer.analyze_file("tests/fixtures/polka_24.mp3")
    assert features['meter'] == 'binary'
```

### Test Fixtures

Audio test fixtures should:
- Be small in size (< 1MB each)
- Use Creative Commons or public domain licenses
- Represent diverse dance styles and recording conditions
- Be documented with ground truth labels

**Adding test fixtures:**
1. Place file in `tests/fixtures/`
2. Create metadata file: `tests/fixtures/metadata.json`
3. Document license and source

```json
{
  "known_120bpm.mp3": {
    "tempo_bpm": 120,
    "meter": "binary",
    "style": "Polka",
    "license": "CC-BY-4.0",
    "source": "Traditional recording"
  }
}
```

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

**Format:**
```
<type>: <short summary>

<detailed description>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```
feat: Add support for 5/4 meter detection

Implemented detection logic for irregular 5/4 meter commonly
found in modern folk fusion. Uses specialized beat tracking
with madmom's DBNDownBeatTracker.

Closes #42
```

```
fix: Correct swing ratio calculation for triplet feels

Previous calculation incorrectly weighted the first beat.
Now properly calculates median IOI ratio as per Butterfield (2006).

Fixes #103
```

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and linting:**
   ```bash
   pytest
   black --check neckenml/ tests/
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: Add feature description"
   ```

5. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request:**
   - Provide clear description of changes
   - Reference related issues
   - Include test results
   - Add screenshots/examples if relevant

**Pull Request Template:**
```markdown
## Description
Brief description of what this PR does.

## Related Issues
Fixes #123
Related to #456

## Changes
- Added X feature
- Fixed Y bug
- Improved Z performance

## Testing
- [ ] Added unit tests
- [ ] All tests pass
- [ ] Tested with sample audio files

## Documentation
- [ ] Updated docstrings
- [ ] Updated README if needed
- [ ] Added examples if needed
```

### Code Review Process

All submissions require code review. Reviewers will check:
- Code follows style guidelines
- Tests are comprehensive and passing
- Documentation is clear and complete
- Changes don't break existing functionality
- Performance impact is acceptable

**Addressing review feedback:**
- Respond to all comments
- Make requested changes in new commits (don't force-push)
- Mark conversations as resolved when addressed

## Community

### Getting Help

- **Documentation:** Start with [docs/](docs/)
- **Issues:** Search existing issues or create new one
- **Discussions:** Use issue discussions for questions

### Recognition

Contributors are recognized in:
- Project README
- Release notes
- Git commit history

Thank you for contributing to Neckenml Analyzer! Your efforts help preserve and promote Swedish folk music culture through technology.
