"""Database models for neckenml analyzer."""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class Track(Base):
    """
    Track model containing basic track information and analysis results.

    This is a minimal, generic model suitable for any music analysis application.
    Applications can extend this model.
    """
    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String, index=True)
    artist: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Analysis metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    analysis_version: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )

    # Relationships
    analysis_sources: Mapped[List["AnalysisSource"]] = relationship(
        "AnalysisSource", back_populates="track"
    )
    dance_styles: Mapped[List["TrackDanceStyle"]] = relationship(
        "TrackDanceStyle", back_populates="track"
    )


class AnalysisSource(Base):
    """
    Stores raw analysis data from different analysis sources.

    The raw_data JSONB field contains expensive-to-compute artifacts that enable
    fast re-analysis without re-processing audio files. This includes:

    Structure:
    {
        "version": "1.0.0",  # Schema version for future migrations

        # MADMOM RNN BEAT DETECTION (most expensive, enables all rhythm re-analysis)
        "rhythm_extractor": {
            "source": "madmom_rnn_downbeat",      # Which algorithm was used
            "bpm": float,                          # Detected tempo
            "beats": [float, ...],                 # Beat positions in seconds
            "beat_positions": [int, ...],          # Beat numbers (1=downbeat, 2/3/4=other)
            "activation_functions": [[float, ...], ...],  # Raw RNN activations (time-series)
            "beats_per_bar": int,                  # Detected meter (3 or 4)
            "ternary_confidence": float,           # Meter confidence (0-1)
            "fps": int,                            # Activation function frame rate
            "beat_info": [[time, beat_num], ...],  # Combined beat data
            "bars": [float, ...]                   # Bar positions in seconds
        },

        # NEURAL NETWORK OUTPUTS
        "musicnn": {
            "raw_embeddings": [[float, ...], ...],  # Time-series embeddings (optional, large)
            "avg_embedding": [float, ...]           # 200-dim averaged embedding
        },

        "vocal": {
            "predictions": [[float, float], ...],  # Time-series [instrumental, vocal] (optional)
            "instrumental_score": float,           # Aggregated instrumental probability
            "vocal_score": float                   # Aggregated vocal probability
        },

        # AUDIO STATISTICS (enables folk authenticity and other derived features)
        "audio_stats": {
            "loudness_lufs": float,
            "rms": float,
            "zcr": float,
            "onset_rate": float,
            "duration_seconds": float
        },

        # SUB-BEAT ANALYSIS (for swing recalculation)
        "onsets": {
            "librosa_onset_times": [float, ...]  # Onset detections from librosa
        },

        # STRUCTURE ANALYSIS
        "structure": {
            "mfcc_matrix": [[float, ...], ...],  # Full MFCC coefficients (optional, large)
            "sections": [float, ...],            # Section boundaries in seconds
            "section_labels": [str, ...]         # Section labels (A, B, etc.)
        },

        # DYNAMICS (for articulation/bounciness recalculation)
        "dynamics": {
            "envelope": [float, ...],        # Amplitude envelope (optional, can be downsampled)
            "beat_activations": [float, ...] # Energy at each beat position
        }
    }

    With these artifacts stored, the following can be quickly re-calculated without audio:
    - Swing ratio, punchiness, polska/hambo scores
    - Articulation, bounciness
    - Folk features, BPM stability
    - New classification models from embeddings
    - Structure refinement with different style hints
    """
    __tablename__ = "analysis_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    track_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tracks.id"))
    source_type: Mapped[str] = mapped_column(String, index=True)
    raw_data: Mapped[dict] = mapped_column(JSONB)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship
    track: Mapped["Track"] = relationship("Track", back_populates="analysis_sources")


class TrackDanceStyle(Base):
    """
    Dance style classification results for a track.

    Stores the classified dance style, confidence, and the feature embedding
    used for classification. Applications can extend this model.
    """
    __tablename__ = "track_dance_styles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    track_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tracks.id"))

    # Classification results
    dance_style: Mapped[str] = mapped_column(String, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Feature embedding used for classification (217-dimensional vector)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(217), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship
    track: Mapped["Track"] = relationship("Track", back_populates="dance_styles")
