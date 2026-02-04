"""
Tunable parameters for the StyleClassifier heuristic rules.

This allows the classifier to be optimized based on user feedback
without hardcoding threshold values.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict
import json
from pathlib import Path


@dataclass
class ClassifierParams:
    """
    Tunable parameters for heuristic dance style classification.

    These parameters control the thresholds and ranges used in the
    heuristic classifier. They can be optimized based on user feedback
    to improve classification accuracy.
    """

    # === HAMBO DETECTION ===
    hambo_score_threshold: float = 0.30
    """Minimum hambo_score to trigger Hambo classification (ternary)"""

    hambo_ratio_threshold: float = 0.40
    """Minimum first beat ratio to classify as Hambo (ternary)"""

    # === SCHOTTIS DETECTION ===
    schottis_swing_threshold: float = 1.35
    """Minimum swing ratio to classify as Schottis (binary)"""

    # === VALS DETECTION ===
    vals_ratio_tolerance: float = 0.08
    """Maximum deviation from 0.33 for even beat detection (ternary)"""

    vals_polska_score_max: float = 0.30
    """Maximum polska_score for strong Vals classification"""

    # === POLSKA DETECTION ===
    polska_score_fallback: float = 0.40
    """Minimum polska_score for fallback Polska classification (ternary)"""

    polska_rescue_ternary_min: float = 0.55
    """Minimum ternary_confidence to rescue misdetected Polska (binary→ternary)"""

    polska_rescue_polska_score_min: float = 0.25
    """Minimum polska_score for Polska rescue when ternary_conf < 0.60"""

    polska_rescue_signals_required: int = 4
    """Number of signals required to rescue misdetected Polska"""

    # === MAZURKA DETECTION ===
    mazurka_swing_high: float = 1.80
    """Swing threshold for strong Mazurka classification (ternary)"""

    mazurka_swing_medium: float = 1.20
    """Swing threshold for Mazurka with beat 2 emphasis (ternary)"""

    # === MENUETT DETECTION ===
    menuett_swing_min: float = 0.70
    """Minimum swing for Menuett classification"""

    menuett_swing_max: float = 1.05
    """Maximum swing for Menuett classification"""

    menuett_bpm_max: float = 115.0
    """Maximum BPM for Menuett (stately tempo)"""

    # === ENGELSKA DETECTION ===
    engelska_swing_min: float = 0.70
    """Minimum swing for Engelska classification (binary)"""

    engelska_swing_max: float = 0.92
    """Maximum swing for Engelska classification (binary)"""

    engelska_bpm_min: float = 115.0
    """Minimum BPM for Engelska (fast tempo)"""

    # === POLKA DETECTION ===
    polka_swing_max: float = 1.10
    """Maximum swing for Polka classification (binary, even beats)"""

    # === SNOA DETECTION ===
    snoa_bpm_min: float = 80.0
    """Minimum BPM for Snoa (walking tempo)"""

    snoa_bpm_max: float = 115.0
    """Maximum BPM for Snoa (walking tempo)"""

    snoa_swing_max: float = 1.20
    """Maximum swing for Snoa (moderate swing)"""

    snoa_swing_fallback_min: float = 0.90
    """Minimum swing for Snoa when BPM unavailable"""

    snoa_swing_fallback_max: float = 1.15
    """Maximum swing for Snoa when BPM unavailable"""

    # === METADATA ===
    version: int = 1
    """Parameter version number (increments with each optimization)"""

    training_samples: int = 0
    """Number of user-confirmed samples used for training these parameters"""

    last_updated: str = ""
    """ISO timestamp of last parameter update"""

    accuracy_on_training: float = 0.0
    """Classification accuracy on training set with these parameters"""

    def to_dict(self) -> Dict:
        """Convert parameters to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ClassifierParams':
        """Create parameters from dictionary."""
        return cls(**data)

    def save(self, path: str):
        """Save parameters to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ClassifierParams':
        """Load parameters from JSON file."""
        if not Path(path).exists():
            return cls()  # Return defaults if file doesn't exist

        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def copy(self) -> 'ClassifierParams':
        """Create a copy of these parameters."""
        return ClassifierParams(**self.to_dict())


# Default global parameters (used if not provided)
_default_params = ClassifierParams()


def get_default_params() -> ClassifierParams:
    """Get default classifier parameters."""
    return _default_params.copy()


def set_default_params(params: ClassifierParams):
    """Set default classifier parameters globally."""
    global _default_params
    _default_params = params.copy()
