"""
Example: Artifact Persistence and Fast Re-analysis

This example demonstrates how to:
1. Analyze audio once and store expensive artifacts
2. Re-analyze multiple times without touching the audio file
3. Update classification logic without re-running audio processing

This is useful for:
- Iterating on classification models
- Tweaking feature engineering logic
- A/B testing different scoring algorithms
- Bulk re-classification when models are updated
"""

from neckenml import AudioAnalyzer, compute_derived_features
from neckenml.sources import FileAudioSource
import json


def main():
    # ============================================================
    # STEP 1: Initial Analysis with Artifact Storage
    # ============================================================
    print("=" * 60)
    print("STEP 1: Analyzing audio and storing artifacts")
    print("=" * 60)

    source = FileAudioSource(audio_dir="/path/to/audio/files")
    analyzer = AudioAnalyzer(audio_source=source)

    # Analyze with return_artifacts=True to get both features and raw data
    result = analyzer.analyze_file(
        file_path="/path/to/audio/file.mp3",
        return_artifacts=True
    )

    # Result now contains:
    # - result["features"]: All derived features (as usual)
    # - result["raw_artifacts"]: Expensive artifacts to store in database

    features = result["features"]
    raw_artifacts = result["raw_artifacts"]

    print(f"\nInitial classification: {features['ml_suggested_style']}")
    print(f"Confidence: {features['ml_confidence']:.2%}")
    print(f"\nStored artifacts size: {len(json.dumps(raw_artifacts))} bytes")

    # ============================================================
    # SIMULATE: Store raw_artifacts in database
    # ============================================================
    # In your application, you would store this in AnalysisSource.raw_data:
    #
    # analysis_source = AnalysisSource(
    #     track_id=track.id,
    #     source_type="neckenml_analyzer",
    #     raw_data=raw_artifacts,
    #     confidence_score=features['ml_confidence']
    # )
    # session.add(analysis_source)
    # session.commit()

    print("\nArtifacts stored in database (simulated)")

    # ============================================================
    # STEP 2: Fast Re-analysis from Stored Artifacts
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Re-analyzing from stored artifacts (NO AUDIO)")
    print("=" * 60)

    # Simulate retrieving artifacts from database
    # In reality: raw_artifacts = analysis_source.raw_data

    # Re-compute all features WITHOUT accessing the audio file
    reanalyzed_features = compute_derived_features(raw_artifacts)

    print(f"\nRe-analyzed classification: {reanalyzed_features['ml_suggested_style']}")
    print(f"Confidence: {reanalyzed_features['ml_confidence']:.2%}")

    # Verify they match
    assert reanalyzed_features['ml_suggested_style'] == features['ml_suggested_style']
    assert abs(reanalyzed_features['ml_confidence'] - features['ml_confidence']) < 0.001

    print("\nRe-analysis matches original (as expected)")

    # ============================================================
    # STEP 3: Re-analysis with Updated Classifier
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Re-analysis with new classifier (still NO AUDIO)")
    print("=" * 60)

    # Imagine you've trained a new model
    # from neckenml.classifier.style_head import ClassificationHead
    # new_classifier = ClassificationHead(model_path="./my_new_model.pkl")

    # Re-compute with new classifier
    # updated_features = compute_derived_features(
    #     raw_artifacts,
    #     new_classifier=new_classifier
    # )

    # print(f"\nNew classifier result: {updated_features['ml_suggested_style']}")
    # print(f"New confidence: {updated_features['ml_confidence']:.2%}")

    print("\n(New classifier example commented out - uncomment to use)")

    # ============================================================
    # STEP 4: Bulk Re-classification Example
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Bulk re-classification from database")
    print("=" * 60)

    # Simulate bulk re-analysis of all tracks
    # In reality, query all AnalysisSource records:
    #
    # from sqlalchemy.orm import Session
    # from neckenml.models import AnalysisSource
    #
    # def bulk_reclassify(session: Session, new_classifier=None):
    #     """Re-classify all tracks without touching audio files."""
    #
    #     analysis_sources = session.query(AnalysisSource).all()
    #
    #     for source in analysis_sources:
    #         # Fast re-analysis from stored artifacts
    #         new_features = compute_derived_features(
    #             source.raw_data,
    #             new_classifier=new_classifier
    #         )
    #
    #         # Update classification in database
    #         # (You could store this in a new TrackDanceStyle record)
    #         print(f"Track {source.track_id}: {new_features['ml_suggested_style']}")
    #
    #     session.commit()
    #
    # # Re-classify 1000 tracks in seconds instead of hours!
    # bulk_reclassify(session, new_classifier=my_improved_model)

    print("\n(Bulk re-classification example commented out)")

    # ============================================================
    # Performance Comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    print("""
Without artifact storage:
  - Change classification logic → Re-run full audio analysis
  - 1000 tracks × 30 seconds = 8.3 hours
  - High CPU/GPU usage (Madmom, MusiCNN)

With artifact storage:
  - Change classification logic → Re-run from artifacts
  - 1000 tracks × 0.1 seconds = 1.7 minutes
  - Low CPU usage (just math on stored vectors)

Speed improvement: ~300x faster
    """)

    # ============================================================
    # What's Stored vs What's Re-computed
    # ============================================================
    print("=" * 60)
    print("WHAT'S STORED vs WHAT'S RE-COMPUTED")
    print("=" * 60)

    print("""
STORED (expensive, rarely changes):
  ✓ Madmom beat detection
  ✓ MusiCNN embeddings
  ✓ Vocal detection (neural network)
  ✓ Librosa onset detection
  ✓ Audio envelope
  ✓ Beat activations
  ✓ Audio statistics (RMS, ZCR, etc.)

RE-COMPUTED (cheap math, may change often):
  ⚡ BPM, swing ratio, punchiness
  ⚡ Polska/Hambo scores
  ⚡ Articulation, bounciness
  ⚡ Full feature vector
  ⚡ ML classification
  ⚡ Folk authenticity score
    """)


if __name__ == "__main__":
    main()
