# Extending neckenml Analyzer

## Custom Audio Sources

The `AudioSource` interface allows you to fetch audio from any source: cloud storage, APIs, databases, etc.

### Example: Cloud Storage Audio Source

```python
import os
from neckenml.sources import AudioSource

class CloudStorageAudioSource(AudioSource):
    """Fetch audio files from cloud object storage."""

    def __init__(self, bucket_name: str, storage_client, prefix: str = 'audio/'):
        self.client = storage_client
        self.bucket = bucket_name
        self.prefix = prefix
        self.temp_dir = "/tmp/neckenml_audio"
        os.makedirs(self.temp_dir, exist_ok=True)

    def fetch_audio(self, track_id: str) -> str:
        """Download audio from cloud storage to temp directory."""
        # Try different extensions
        for ext in ['mp3', 'wav', 'flac']:
            object_key = f"{self.prefix}{track_id}.{ext}"
            local_path = os.path.join(self.temp_dir, f"{track_id}.{ext}")

            try:
                self.client.download_file(
                    bucket=self.bucket,
                    key=object_key,
                    destination=local_path
                )
                return local_path
            except FileNotFoundError:
                continue

        raise FileNotFoundError(f"Audio not found: {self.prefix}{track_id}.*")

    def cleanup(self, file_path: str) -> None:
        """Delete temporary file."""
        if os.path.exists(file_path):
            os.remove(file_path)

# Usage
from neckenml import AudioAnalyzer

source = CloudStorageAudioSource(bucket_name='my-music-bucket', storage_client=my_client)
analyzer = AudioAnalyzer(audio_source=source)

# Analyze will automatically fetch and clean up
features = analyzer.analyze('track123')
```

### Example: HTTP Audio Source

```python
import requests
from neckenml.sources import AudioSource

class HTTPAudioSource(AudioSource):
    """Fetch audio from HTTP URLs."""

    def __init__(self, base_url: str, temp_dir: str = "/tmp"):
        self.base_url = base_url.rstrip('/')
        self.temp_dir = temp_dir

    def fetch_audio(self, track_id: str) -> str:
        """Download audio via HTTP."""
        url = f"{self.base_url}/{track_id}.mp3"
        local_path = f"{self.temp_dir}/{track_id}.mp3"

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return local_path

    def cleanup(self, file_path: str) -> None:
        if os.path.exists(file_path):
            os.remove(file_path)
```

## Training Custom Models

### Training from Database

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from neckenml.training import TrainingService

engine = create_engine('postgresql://localhost/neckenml')
Session = sessionmaker(bind=engine)
db = Session()

trainer = TrainingService()

# Train on all data
trainer.train_from_database(db)

# Or train on specific tracks
selected_tracks = [1, 2, 3, 5, 8, 13]  # Your curated training set
trainer.train_from_database(db, track_ids=selected_tracks)
```

### Training from CSV

Prepare a CSV file with embeddings and labels:

```csv
embedding,style
"[0.1, 0.2, 0.3, ...]","Polska"
"[0.4, 0.5, 0.6, ...]","Hambo"
```

Then train:

```python
from neckenml.training import TrainingService

trainer = TrainingService()
trainer.train_from_csv('training_data.csv')
```

### Training from Raw Data

```python
import numpy as np
from neckenml.training import TrainingService

# Collect your training data
embeddings = []
labels = []

for audio_file, label in your_labeled_dataset:
    features = analyzer.analyze_file(audio_file)
    embeddings.append(features['embedding'])
    labels.append(label)

# Train
trainer = TrainingService()
trainer.train_from_data(
    embeddings=np.array(embeddings),
    labels=labels
)
```

## Extending Database Models

You can extend the base models with custom fields:

```python
from sqlalchemy import Column, String, Boolean, Integer
from neckenml.models.schema import Track as BaseTrack
from neckenml.models.schema import TrackDanceStyle as BaseTrackDanceStyle

class Track(BaseTrack):
    """Extended track with custom fields."""
    __tablename__ = 'tracks'  # Same table name

    # Add custom field
    my_custom_field = Column(String)

class TrackDanceStyle(BaseTrackDanceStyle):
    """Extended with user feedback tracking."""
    __tablename__ = 'track_dance_styles'

    # Add feedback fields
    is_user_confirmed = Column(Boolean, default=False)
    confirmation_count = Column(Integer, default=0)
    user_notes = Column(String)
```

## Custom Classification Logic

Extend the StyleClassifier with your own heuristics:

```python
from neckenml import StyleClassifier

class CustomStyleClassifier(StyleClassifier):
    """Classifier with custom rules for my region."""

    def classify(self, features: dict) -> dict:
        # First, try parent classifier
        result = super().classify(features)

        # Add custom logic
        if features['tempo_bpm'] > 180 and features['meter'] == '2/4':
            result['primary_style'] = 'FastPolka'
            result['confidence'] = 0.95
            result['source'] = 'custom_heuristic'

        return result
```

## Batch Processing

Process many files efficiently:

```python
from concurrent.futures import ThreadPoolExecutor
from neckenml import AudioAnalyzer, StyleClassifier
from neckenml.sources import FileAudioSource

source = FileAudioSource(audio_dir="/music/library")
analyzer = AudioAnalyzer(audio_source=source)
classifier = StyleClassifier()

def process_track(track_id):
    try:
        features = analyzer.analyze(track_id)
        result = classifier.classify(features)
        return (track_id, result['primary_style'], result['confidence'])
    except Exception as e:
        return (track_id, 'ERROR', 0.0)

# Process in parallel
track_ids = ["song1", "song2", "song3", ...]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_track, track_ids))

for track_id, style, confidence in results:
    print(f"{track_id}: {style} ({confidence:.0%})")
```

## Model Versioning

When you update feature extraction (change FEATURE_VERSION), old models become incompatible:

```python
from neckenml.classifier import ClassificationHead

# Check current version
head = ClassificationHead()
print(f"Feature version: {head.FEATURE_VERSION}")
print(f"Expected features: {head.EXPECTED_FEATURE_COUNT}")

# If version mismatch, you'll need to retrain
if head.model is None:
    print("[WARNING] Model version mismatch - retraining needed")
```
