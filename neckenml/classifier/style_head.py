import os
from typing import Optional


class ClassificationHead:
    """
    The 'Head' of the model.
    It sits on top of the MusiCNN 'Backbone' (Embedding Extractor).
    It learns to map Embeddings -> Folk Styles based on user feedback.
    """

    # Increment this when changing feature vector composition
    FEATURE_VERSION = 4

    EXPECTED_FEATURE_COUNT = 217

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the classification head.

        Args:
            model_path: Path to the trained model file. If None, tries:
                1. Package's bundled pre-trained model
                2. ~/.neckenml/custom_style_head.pkl
        """
        if model_path is None:
            # Try package's bundled model first
            package_model = os.path.join(
                os.path.dirname(__file__), "..", "models", "pretrained", "custom_style_head.pkl"
            )
            if os.path.exists(package_model):
                self.model_path = package_model
            else:
                # Fall back to user's home directory
                self.model_path = os.path.expanduser("~/.neckenml/custom_style_head.pkl")
        else:
            self.model_path = model_path

        self.scaler = None
        self.model = None
        self.model_version = None
        self._load()

    def _load(self):
        import joblib
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_version = data.get('version', 0)
            
            # Version mismatch check
            if self.model_version != self.FEATURE_VERSION:
                print(f"[WARNING] Model version mismatch! Model: v{self.model_version}, Expected: v{self.FEATURE_VERSION}")
                print("Clearing model - will need retraining.")
                self.model = None
                self.scaler = None
            else:
                print("Classification Head loaded.")
        else:
            print("New Classification Head initialized (Waiting for training data).")

    def predict(self, embedding: list) -> tuple[str, float]:
        """
        Returns (style_string, confidence) tuple.
        E.g., ('Hambo', 0.85) or ('Unknown', 0.0)
        """
        import numpy as np

        if self.model is None or self.scaler is None:
            return ("Unknown", 0.0)
        
        try:
            # --- SAFETY CHECK FOR HYBRID UPDATE ---
            # The model expects a specific number of features.
            expected_features = self.scaler.n_features_in_
            if len(embedding) > expected_features:
                embedding = embedding[:expected_features]
            elif len(embedding) < expected_features:
                 return ("Unknown", 0.0)

            # 1. Normalize
            scaled_features = self.scaler.transform([embedding])
            
            # 2. Predict
            probs = self.model.predict_proba(scaled_features)
            max_prob = float(np.max(probs))
            
            if max_prob < 0.4: 
                return ("Unknown", max_prob)
                
            prediction_idx = np.argmax(probs)
            style = self.model.classes_[prediction_idx]
            return (style, max_prob)
        except Exception as e:
            print(f"Head prediction failed: {e}")
            return ("Unknown", 0.0)
    
    def train(self, embeddings: list, labels: list):
        """
        Retrains the head using the provided Golden Dataset.
        """
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        if not embeddings or not labels:
            print("No data to train on.")
            return
        
        valid_embeddings = []
        valid_labels = []

        print(f"Validating {len(embeddings)} samples against expected size {self.EXPECTED_FEATURE_COUNT}...")

        for emb, lbl in zip(embeddings, labels):
            # Only keep vectors that match the current version (217)
            if len(emb) == self.EXPECTED_FEATURE_COUNT:
                valid_embeddings.append(emb)
                valid_labels.append(lbl)
        
        if len(valid_embeddings) == 0:
            print(f"[WARNING] No valid samples found! All {len(embeddings)} inputs were the wrong size.")
            print("   Action: You need to re-analyze tracks to generate new 217-len vectors.")
            return

        print(f"Training Head on {len(labels)} examples...")
        
        X = np.array(embeddings)
        y = np.array(labels)

        # 1. Normalize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 2. Fit Model
        # RandomForest is robust against noise and works well with embeddings
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)

        # 3. Save
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'version': self.FEATURE_VERSION
        }, self.model_path)
        print("Classification Head saved to disk.")