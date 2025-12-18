"""Database models for neckenml analyzer."""

import uuid
from datetime import datetime
from typing import List
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
    artist: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Analysis metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    analysis_version: Mapped[str | None] = mapped_column(
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

    The raw_data JSONB field contains the complete analysis results,
    including embeddings, features, and metadata.
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
    embedding: Mapped[list[float] | None] = mapped_column(Vector(217), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship
    track: Mapped["Track"] = relationship("Track", back_populates="dance_styles")
