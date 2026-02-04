"""Training service for the classification head."""

import numpy as np
from typing import List, Optional
from sqlalchemy.orm import Session
from neckenml.core.classifier.style_head import ClassificationHead


class TrainingService:
    """
    Service for training the dance style classifier.

    Provides flexible training from various data sources:
    - Direct embeddings and labels (train_from_data)
    - Database records (train_from_database)
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the training service.

        Args:
            model_path: Path where the trained model will be saved.
                       If None, uses ClassificationHead's default location.
        """
        self.head = ClassificationHead(model_path=model_path)

    def train_from_data(self, embeddings: np.ndarray, labels: List[str]) -> bool:
        """
        Train classifier from embeddings and labels.

        Args:
            embeddings: Nx217 array of feature vectors from AudioAnalyzer
            labels: List of dance style names (e.g., ['Polska', 'Hambo', ...])

        Returns:
            bool: True if training succeeded, False otherwise

        Example:
            >>> trainer = TrainingService()
            >>> embeddings = np.array([[...], [...]])  # Nx217 features
            >>> labels = ['Polska', 'Hambo']
            >>> trainer.train_from_data(embeddings, labels)
        """
        if len(embeddings) == 0 or len(labels) == 0:
            print("No training data provided")
            return False

        if len(embeddings) != len(labels):
            print(f"[WARNING] Mismatch: {len(embeddings)} embeddings but {len(labels)} labels")
            return False

        # Minimum samples needed for a useful classifier
        if len(embeddings) < 5:
            print(f"Need at least 5 samples to train, got {len(embeddings)}")
            return False

        # Convert to list if numpy array
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        print(f"Training classifier on {len(labels)} examples...")
        self.head.train(embeddings, labels)
        return True

    def train_from_database(
        self,
        db_session: Session,
        track_ids: Optional[List[int]] = None
    ) -> bool:
        """
        Train from TrackDanceStyle records in the database.

        This is a generic database training method

        Args:
            db_session: SQLAlchemy database session
            track_ids: Optional list of track IDs to train on.
                      If None, trains on all tracks with analysis data.

        Returns:
            bool: True if training succeeded, False otherwise

        Example:
            >>> from sqlalchemy import create_engine
            >>> from sqlalchemy.orm import sessionmaker
            >>> engine = create_engine('postgresql://...')
            >>> Session = sessionmaker(bind=engine)
            >>> db = Session()
            >>> trainer = TrainingService()
            >>> # Train on specific tracks
            >>> trainer.train_from_database(db, track_ids=[1, 2, 3])
            >>> # Or train on all available data
            >>> trainer.train_from_database(db)
        """
        from neckenml.core.models.schema import Track, TrackDanceStyle, AnalysisSource

        print("Querying database for training data...")

        # Build query
        query = (
            db_session.query(TrackDanceStyle, AnalysisSource)
            .join(Track, TrackDanceStyle.track_id == Track.id)
            .join(AnalysisSource, AnalysisSource.track_id == Track.id)
            .filter(AnalysisSource.source_type == 'hybrid_ml_v2')
        )

        # Filter by specific track IDs if provided
        if track_ids is not None:
            query = query.filter(Track.id.in_(track_ids))

        results = query.all()

        if not results:
            print("[WARNING] No training data found in database")
            return False

        # Extract embeddings and labels
        embeddings = []
        labels = []

        for style_row, analysis in results:
            emb = analysis.raw_data.get('embedding')
            if emb and len(emb) == self.head.EXPECTED_FEATURE_COUNT:
                embeddings.append(emb)
                labels.append(style_row.dance_style)

        if not embeddings:
            print("No valid embeddings found (check FEATURE_VERSION)")
            return False

        print(f"Found {len(embeddings)} valid training samples")

        # Train using the extracted data
        return self.train_from_data(np.array(embeddings), labels)

    def train_from_csv(self, csv_path: str, embedding_col: str = 'embedding', label_col: str = 'style') -> bool:
        """
        Train from a CSV file containing embeddings and labels.

        Args:
            csv_path: Path to CSV file
            embedding_col: Name of column containing embeddings (as JSON array or comma-separated)
            label_col: Name of column containing style labels

        Returns:
            bool: True if training succeeded

        Example CSV format:
            embedding,style
            "[0.1, 0.2, ...]","Polska"
            "[0.3, 0.4, ...]","Hambo"
        """
        import pandas as pd
        import json

        print(f"Loading training data from {csv_path}...")

        try:
            df = pd.read_csv(csv_path)

            if embedding_col not in df.columns or label_col not in df.columns:
                print(f"[WARNING] CSV must have '{embedding_col}' and '{label_col}' columns")
                return False

            embeddings = []
            labels = []

            for _, row in df.iterrows():
                # Parse embedding (handle JSON string or list)
                emb = row[embedding_col]
                if isinstance(emb, str):
                    emb = json.loads(emb)

                if len(emb) == self.head.EXPECTED_FEATURE_COUNT:
                    embeddings.append(emb)
                    labels.append(row[label_col])

            return self.train_from_data(np.array(embeddings), labels)

        except Exception as e:
            print(f"[ERROR] Error loading CSV: {e}")
            return False
