"""
Folk Authenticity Detector

Analyzes audio features to determine if a track is genuinely traditional folk music
or if it's modern/electronic/pop music. This complements the metadata-based genre
classifier by examining actual audio characteristics.

Traditional folk music typically has:
- Acoustic instrumentation (no synths/electronic elements)
- Natural dynamics (human-performed, not heavily compressed)
- Organic timing (human imperfections, not quantized)
- Simpler production (less processing, natural reverb)
- Natural vocal qualities (if present)

Note: This module is MIT licensed and does not depend on any AGPL code.
It operates on pre-computed features, not raw audio.
"""

import numpy as np


class FolkAuthenticityDetector:
    """
    Detects whether audio has characteristics of traditional folk music
    vs modern/electronic production.

    Returns a confidence score from 0.0 (definitely modern) to 1.0 (definitely traditional folk).
    """

    def __init__(self, manual_review_threshold: float = 0.6):
        """
        Args:
            manual_review_threshold: Scores below this require human review
        """
        self.manual_review_threshold = manual_review_threshold

    def analyze(
        self,
        rms_value: float,
        zcr_value: float,
        swing_ratio: float,
        articulation: str,
        bounciness: float,
        voice_probability: float,
        is_likely_instrumental: bool,
        embedding
    ) -> dict:
        """
        Analyze audio features to determine folk authenticity.

        Args:
            rms_value: RMS amplitude (dynamics indicator)
            zcr_value: Zero crossing rate (timbral complexity)
            swing_ratio: Timing swing ratio
            articulation: "smooth", "punchy", "staccato"
            bounciness: How bouncy the rhythm feels
            voice_probability: Confidence of vocal presence
            is_likely_instrumental: Whether track is instrumental
            embedding: MusiCNN embedding vector

        Returns:
            dict with:
                - folk_authenticity_score: 0.0-1.0
                - requires_manual_review: bool
                - confidence_breakdown: dict of individual signals
        """

        scores = {}

        # Signal 1: Dynamic Range (Traditional folk has wider dynamics)
        # RMS values closer to 0 suggest more dynamic variation
        # Modern music is heavily compressed (high RMS, low variation)
        dynamic_score = self._score_dynamics(rms_value)
        scores['dynamics'] = dynamic_score

        # Signal 2: Timbral Complexity (Acoustic vs Electronic)
        # Traditional folk has moderate ZCR (acoustic instruments)
        # Electronic music has very high or very low ZCR
        timbre_score = self._score_timbre(zcr_value)
        scores['timbre'] = timbre_score

        # Signal 3: Human Timing (Swing & Groove)
        # Traditional folk has natural swing (not perfectly quantized)
        # Swing ratio between 0.55-0.70 is very human
        timing_score = self._score_timing(swing_ratio, bounciness)
        scores['timing'] = timing_score

        # Signal 4: Articulation Style
        # Traditional folk tends to be "smooth" or moderately "punchy"
        # Heavy "staccato" or extreme punchiness suggests modern production
        articulation_score = self._score_articulation(articulation)
        scores['articulation'] = articulation_score

        # Signal 5: Vocal Quality (if vocals present)
        # Traditional folk has natural, unprocessed vocals
        # This is harder to detect, so we give it lower weight
        if not is_likely_instrumental and voice_probability > 0.3:
            vocal_score = self._score_vocals(voice_probability)
            scores['vocals'] = vocal_score
        else:
            scores['vocals'] = 0.5  # Neutral for instrumentals

        # Signal 6: Embedding-based Production Detection
        # MusiCNN embeddings can capture production characteristics
        # We look at the variance and energy distribution
        production_score = self._score_production(embedding)
        scores['production'] = production_score

        # Weighted average (emphasize most reliable signals)
        weights = {
            'dynamics': 0.20,
            'timbre': 0.20,
            'timing': 0.15,
            'articulation': 0.15,
            'vocals': 0.10,
            'production': 0.20
        }

        final_score = sum(scores[k] * weights[k] for k in weights.keys())

        # Clamp to 0-1 range
        final_score = max(0.0, min(1.0, final_score))

        return {
            'folk_authenticity_score': final_score,
            'requires_manual_review': final_score < self.manual_review_threshold,
            'confidence_breakdown': scores,
            'interpretation': self._interpret_score(final_score)
        }

    def _score_dynamics(self, rms: float) -> float:
        """
        Score based on dynamic range.
        Traditional folk: 0.4-0.7 (moderate dynamics)
        Modern compressed: 0.8+ (very loud, no dynamics)
        """
        if rms < 0.3:
            return 0.6  # Too quiet, maybe amateur recording
        elif 0.3 <= rms < 0.7:
            return 0.9  # Good natural dynamics
        elif 0.7 <= rms < 0.85:
            return 0.5  # Moderate compression
        else:
            return 0.2  # Heavily compressed (modern production)

    def _score_timbre(self, zcr: float) -> float:
        """
        Score based on zero crossing rate.
        Acoustic instruments: 0.05-0.15
        Electronic/Synth: Very high or very low
        """
        if 0.05 <= zcr <= 0.15:
            return 0.9  # Acoustic sweet spot
        elif 0.03 <= zcr <= 0.20:
            return 0.7  # Likely acoustic
        elif 0.20 < zcr <= 0.30:
            return 0.4  # Possibly electronic elements
        else:
            return 0.2  # Likely electronic/synthetic

    def _score_timing(self, swing_ratio: float, bounciness: float) -> float:
        """
        Score based on human timing characteristics.
        Human swing: 0.55-0.70 (natural triplet feel)
        Quantized: exactly 0.5 or 0.6666...
        """
        score = 0.5

        # Check swing ratio
        if 0.55 <= swing_ratio <= 0.70:
            score += 0.3  # Natural human swing
        elif abs(swing_ratio - 0.5) < 0.02 or abs(swing_ratio - 0.6666) < 0.02:
            score -= 0.2  # Suspiciously perfect quantization

        # Check bounciness (traditional folk is moderately bouncy)
        if 0.3 <= bounciness <= 0.7:
            score += 0.2  # Good natural bounce
        elif bounciness > 0.8:
            score -= 0.1  # Too mechanical

        return max(0.0, min(1.0, score))

    def _score_articulation(self, articulation: str) -> float:
        """
        Score based on articulation style.
        Traditional folk: smooth, punchy
        Modern electronic: heavily staccato, hyper-processed
        """
        if articulation == "smooth":
            return 0.8  # Very typical for folk
        elif articulation == "punchy":
            return 0.6  # Can be folk, but also modern
        elif articulation == "staccato":
            return 0.3  # More typical of electronic music
        else:
            return 0.5  # Unknown

    def _score_vocals(self, voice_probability: float) -> float:
        """
        Score vocal characteristics.
        High confidence vocals in folk context = natural singing
        Very processed vocals would show different patterns (harder to detect)
        """
        # For now, assume if vocals are clearly present, it's a slight indicator of folk
        # (since modern pop often heavily processes vocals, making them harder to detect clearly)
        if voice_probability > 0.7:
            return 0.6  # Clear natural vocals
        elif voice_probability > 0.4:
            return 0.5  # Moderate vocals
        else:
            return 0.5  # Uncertain

    def _score_production(self, embedding) -> float:
        """
        Score based on MusiCNN embedding characteristics.
        Traditional folk: More variance in mid-range frequencies, natural energy distribution
        Modern electronic: Concentrated energy, synthetic patterns
        """
        try:
            if len(embedding) < 200:
                return 0.5  # Not enough data

            # Calculate variance in embedding space
            # Traditional music has more organic variation
            variance = np.var(embedding)

            # Calculate energy concentration
            # Modern music tends to have more concentrated energy
            sorted_vals = np.sort(np.abs(embedding))[::-1]
            top_10_energy = np.sum(sorted_vals[:10]) / np.sum(sorted_vals)

            score = 0.5

            # Higher variance = more organic
            if variance > 0.5:
                score += 0.3
            elif variance > 0.3:
                score += 0.1
            else:
                score -= 0.1

            # Less concentrated = more natural
            if top_10_energy < 0.3:
                score += 0.2
            elif top_10_energy > 0.5:
                score -= 0.2

            return max(0.0, min(1.0, score))

        except Exception:
            return 0.5  # Fallback to neutral

    def _interpret_score(self, score: float) -> str:
        """
        Human-readable interpretation of the score.
        """
        if score >= 0.8:
            return "Very likely traditional folk"
        elif score >= 0.6:
            return "Likely traditional folk"
        elif score >= 0.4:
            return "Uncertain - may be modern folk or folk-pop"
        elif score >= 0.2:
            return "Likely modern/electronic production"
        else:
            return "Very likely modern/electronic music"
