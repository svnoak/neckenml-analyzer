# Installation Guide

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 12+ with pgvector extension
- 4GB+ RAM (for audio analysis)

## Step 1: Install the Package

```bash
pip install neckenml-analyzer
```

Or for development/local installation:

```bash
git clone <repository-url>
cd neckenml-analyzer
pip install -e .
```

## Step 2: Set Up PostgreSQL

### Install PostgreSQL with pgvector

**macOS:**
```bash
brew install postgresql pgvector
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-14-pgvector
```

### Create Database

```bash
# Create database
createdb neckenml

# Enable pgvector extension
psql neckenml -c "CREATE EXTENSION vector;"
```

### Run Migrations

```python
from sqlalchemy import create_engine
from neckenml.models.schema import Base

# Create engine
engine = create_engine('postgresql://localhost/neckenml')

# Create all tables
Base.metadata.create_all(engine)
```

## Step 3: Download Pre-trained Models

The analyzer requires Essentia's MusiCNN models. These are NOT included in the package due to licensing.

### Create Models Directory

```bash
mkdir -p ~/.neckenml/models
```

### Download Models

```bash
# MusiCNN embedding model (3.0 MB)
wget https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb \
  -O ~/.neckenml/models/msd-musicnn-1.pb

# Voice/Instrumental classifier (3.1 MB)
wget https://essentia.upf.edu/models/audio-event-recognition/voice_instrumental/voice_instrumental-musicnn-msd-1.pb \
  -O ~/.neckenml/models/voice_instrumental-musicnn-msd-1.pb
```

### Verify Installation

```bash
ls -lh ~/.neckenml/models/
```

You should see:
```
msd-musicnn-1.pb (3.0 MB)
voice_instrumental-musicnn-msd-1.pb (3.1 MB)
```

## Step 4: Test Installation

```python
from neckenml import AudioAnalyzer, StyleClassifier
from neckenml.sources import FileAudioSource

# This should load without errors
print("neckenml Analyzer installed successfully!")

# Test model loading
analyzer = AudioAnalyzer(model_dir="~/.neckenml/models")
classifier = StyleClassifier()

print("Models loaded successfully!")
```

## Troubleshooting

### ModuleNotFoundError: No module named 'essentia'

```bash
pip install essentia-tensorflow
```

### PostgreSQL Connection Error

Check your connection string:
```python
# Test connection
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost/neckenml')
engine.connect()
```

### Model Files Not Found

Ensure models are in the correct location:
```bash
ls ~/.neckenml/models/msd-musicnn-1.pb
ls ~/.neckenml/models/voice_instrumental-musicnn-msd-1.pb
```

### Memory Errors During Analysis

Audio analysis requires significant RAM. Increase available memory or reduce batch size.

## Optional: Development Dependencies

For development and testing:

```bash
pip install neckenml-analyzer[dev]
```

This installs:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
