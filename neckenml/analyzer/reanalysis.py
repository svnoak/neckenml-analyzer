"""
Re-analysis functions that compute derived features from stored artifacts
without requiring access to the original audio files.

This enables fast iteration on classification logic, feature engineering,
and model updates without re-running expensive audio analysis.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_derived_features(raw_artifacts: Dict[str, Any],
                            metadata_context: str = "",
                            new_classifier=None) -> Dict[str, Any]:
    """
    Computes all derived features from stored raw artifacts.

    This function reproduces the full feature set without accessing audio files,
    enabling fast re-analysis when:
    - Classification models are updated
    - Feature engineering logic changes
    - New scoring algorithms are developed

    Args:
        raw_artifacts: The raw_data dict from AnalysisSource containing:
            - madmom: beat_times, bars, ternary_confidence
            - musicnn: avg_embedding
            - vocal: instrumental_score, vocal_score
            - audio_stats: loudness_lufs, rms, zcr, onset_rate
            - onsets: librosa_onset_times
            - dynamics: envelope, beat_activations
        metadata_context: Optional metadata for classification hints
        new_classifier: Optional custom classifier (defaults to ClassificationHead)

    Returns:
        Dict containing all derived features, matching the output of analyze_file()
    """
    from neckenml.classifier.style_head import ClassificationHead
    from neckenml.analyzer.folk_authenticity import FolkAuthenticityDetector

    # Load classifier
    classifier = new_classifier or ClassificationHead()
    folk_detector = FolkAuthenticityDetector(manual_review_threshold=0.6)

    # Extract stored artifacts
    rhythm = raw_artifacts.get("rhythm_extractor", {})
    musicnn = raw_artifacts.get("musicnn", {})
    vocal = raw_artifacts.get("vocal", {})
    audio_stats = raw_artifacts.get("audio_stats", {})
    onsets_data = raw_artifacts.get("onsets", {})
    dynamics = raw_artifacts.get("dynamics", {})

    # Core rhythm data (from Madmom RNN)
    beat_times = np.array(rhythm.get("beats", []))
    bars = rhythm.get("bars", [])
    ternary_conf = rhythm.get("ternary_confidence", 0.5)
    avg_embedding = np.array(musicnn.get("avg_embedding", []))

    # Vocal
    vocal_score = vocal.get("vocal_score", 0.0)
    instrumental_score = vocal.get("instrumental_score", 1.0)
    is_instrumental = instrumental_score > vocal_score

    # Audio stats
    loudness = audio_stats.get("loudness_lufs", -14.0)
    rms = audio_stats.get("rms", 0.0)
    zcr = audio_stats.get("zcr", 0.0)
    onset_rate = audio_stats.get("onset_rate", 0.0)

    # Dynamics
    beat_activations = dynamics.get("beat_activations", [])
    intervals = dynamics.get("intervals", [])

    # --- RECOMPUTE FOLK FEATURES FROM STORED ARTIFACTS ---
    folk_features = _recompute_folk_features(
        beat_times=beat_times,
        beat_activations=beat_activations,
        intervals=intervals
    )

    # --- RECOMPUTE SWING FROM STORED ONSETS ---
    swing_ratio = _recompute_swing_ratio(
        beat_times=beat_times,
        onset_times=onsets_data.get("librosa_onset_times", [])
    )

    # --- RECOMPUTE FEEL FROM STORED ENVELOPE ---
    articulation, bounciness = _recompute_feel(
        envelope=dynamics.get("envelope", []),
        swing_ratio=swing_ratio
    )

    # --- BUILD FULL FEATURE VECTOR ---
    layout_stats = [rms, zcr, onset_rate]

    folk_vector_list = [
        folk_features["bpm"],
        folk_features["avg_ibi"],
        folk_features["punchiness"],
        folk_features["r1_mean"],
        folk_features["r2_mean"],
        folk_features["r3_mean"],
        folk_features["polska_score"],
        folk_features["hambo_score"],
        folk_features["bpm_stability"]
    ]

    full_vector = np.concatenate([
        avg_embedding,      # 200
        folk_vector_list,   # 9
        [swing_ratio],      # 1
        layout_stats,       # 3
        [ternary_conf],     # 1
        [vocal_score],      # 1
        [articulation],     # 1
        [bounciness]        # 1
    ])

    # --- PREDICT STYLE ---
    predicted_style, ml_confidence = classifier.predict(full_vector)

    # --- FOLK AUTHENTICITY ---
    folk_auth_result = folk_detector.analyze(
        rms_value=rms,
        zcr_value=zcr,
        swing_ratio=swing_ratio,
        articulation=articulation,
        bounciness=bounciness,
        voice_probability=vocal_score,
        is_likely_instrumental=is_instrumental,
        embedding=avg_embedding
    )

    # --- STRUCTURE (from artifacts) ---
    structure = raw_artifacts.get("structure", {})
    sections = structure.get("sections", [])
    section_labels = structure.get("section_labels", [])

    # Get meter
    beat_info = np.array(rhythm.get("beat_info", []))
    meter_numerator = int(np.max(beat_info[:, 1])) if len(beat_info) > 0 else 0

    # --- BUILD RESULT ---
    result = {
        "ml_suggested_style": predicted_style,
        "ml_confidence": float(ml_confidence),
        "embedding": full_vector.tolist(),
        "loudness_lufs": loudness,
        "tempo_bpm": folk_features["bpm"],
        "bpm_stability": folk_features["bpm_stability"],
        "is_likely_instrumental": bool(is_instrumental),
        "voice_probability": float(vocal_score),
        "swing_ratio": float(swing_ratio),
        "articulation": float(articulation),
        "bounciness": float(bounciness),
        "avg_beat_ratios": [
            folk_features["r1_mean"],
            folk_features["r2_mean"],
            folk_features["r3_mean"]
        ],
        "punchiness": folk_features["punchiness"],
        "polska_score": folk_features["polska_score"],
        "hambo_score": folk_features["hambo_score"],
        "ternary_confidence": float(ternary_conf),
        "meter": f"{meter_numerator}/4",
        "bars": [float(b) for b in bars],
        "beat_times": [float(b) for b in beat_times],
        "sections": sections,
        "section_labels": section_labels,
        "folk_authenticity_score": float(folk_auth_result['folk_authenticity_score']),
        "requires_manual_review": bool(folk_auth_result['requires_manual_review']),
        "folk_authenticity_breakdown": folk_auth_result['confidence_breakdown'],
        "folk_authenticity_interpretation": folk_auth_result['interpretation']
    }

    return result


def _recompute_folk_features(beat_times, beat_activations, intervals):
    """Recompute folk features from stored beat data."""
    if len(beat_times) < 12:
        return {
            "bpm": 0.0, "avg_ibi": 0.0, "punchiness": 0.0,
            "r1_mean": 0.33, "r2_mean": 0.33, "r3_mean": 0.34,
            "polska_score": 0.0, "hambo_score": 0.0, "bpm_stability": 0.0
        }

    # Use stored intervals if available, otherwise compute
    if intervals:
        ibis = np.array(intervals)
    else:
        ibis = np.diff(beat_times)

    avg_ibi = np.mean(ibis)
    if avg_ibi == 0:
        return {
            "bpm": 0.0, "avg_ibi": 0.0, "punchiness": 0.0,
            "r1_mean": 0.33, "r2_mean": 0.33, "r3_mean": 0.34,
            "polska_score": 0.0, "hambo_score": 0.0, "bpm_stability": 0.0
        }

    bpm = 60.0 / avg_ibi
    bpm_stability = 1.0 - (np.std(ibis) / avg_ibi)

    # Ratios
    ratios_1, ratios_2, ratios_3 = [], [], []
    triplet_variances = []

    for i in range(0, len(ibis)-2, 3):
        total = np.sum(ibis[i:i+3])
        if total > 0:
            r = ibis[i:i+3] / total
            ratios_1.append(r[0])
            ratios_2.append(r[1])
            ratios_3.append(r[2])
            triplet_variances.append(np.var(r))

    r1_mean = np.mean(ratios_1) if ratios_1 else 0.33
    r2_mean = np.mean(ratios_2) if ratios_2 else 0.33
    r3_mean = np.mean(ratios_3) if ratios_3 else 0.34

    # Punchiness from stored activations
    if beat_activations:
        punchiness = np.tanh((np.sum(beat_activations) / len(beat_activations)) * 10)
    else:
        punchiness = 0.0

    # Polska/Hambo scores
    polska_score, hambo_score = _calculate_ternary_signatures(
        ratios=[r1_mean, r2_mean, r3_mean],
        triplet_variances=triplet_variances,
        intervals=ibis,
        activations=beat_activations
    )

    return {
        "bpm": float(bpm),
        "avg_ibi": float(avg_ibi),
        "punchiness": float(punchiness),
        "r1_mean": float(r1_mean),
        "r2_mean": float(r2_mean),
        "r3_mean": float(r3_mean),
        "polska_score": float(polska_score),
        "hambo_score": float(hambo_score),
        "bpm_stability": float(bpm_stability)
    }


def _calculate_ternary_signatures(ratios, triplet_variances, intervals, activations):
    """Calculate polska/hambo signature scores."""
    r1, r2, r3 = ratios

    timing_variance = np.mean(triplet_variances) if triplet_variances else 0.0
    interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0.0

    # Activation Analysis
    downbeat_dominance = 0.33
    if activations and len(activations) >= 6:
        activations_arr = np.array(activations)
        avg_b1 = np.mean(activations_arr[0::3])
        total = np.mean(activations_arr) * 3
        if total > 0:
            downbeat_dominance = avg_b1 / total

    # Polska score
    polska_score = 0.0
    if (r3 > r1 and r3 > 0.34) or (r2 > r1 and r2 > 0.36):
        polska_score += 0.35
    if timing_variance > 0.003:
        polska_score += min(0.25, timing_variance * 30)
    if downbeat_dominance < 0.38:
        polska_score += 0.15

    # Hambo score
    hambo_score = 0.0
    if r1 > 0.38:
        hambo_score += 0.30 + min(0.20, (r1 - 0.38) * 2)
    if timing_variance < 0.004:
        hambo_score += 0.20
    if downbeat_dominance > 0.40:
        hambo_score += 0.20

    return min(1.0, polska_score), min(1.0, hambo_score)


def _recompute_swing_ratio(beat_times, onset_times):
    """Recompute swing ratio from stored onset times."""
    if not onset_times or len(beat_times) < 2:
        return 1.0

    all_ratios = []

    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]
        beat_duration = beat_end - beat_start

        # Find onsets in the middle of this beat interval
        candidates = [
            o for o in onset_times
            if (beat_start + beat_duration * 0.2) < o < (beat_end - beat_duration * 0.2)
        ]

        if candidates:
            mid_point = min(candidates, key=lambda x: abs(x - (beat_start + (beat_duration / 2))))
            first_half = mid_point - beat_start
            second_half = beat_end - mid_point
            if second_half > 0.001:
                all_ratios.append(first_half / second_half)

    if not all_ratios:
        return 1.0

    return float(np.median(all_ratios))


def _recompute_feel(envelope, swing_ratio):
    """Recompute articulation and bounciness from stored envelope."""
    if not envelope or len(envelope) == 0:
        return 0.0, 0.0

    envelope_arr = np.array(envelope)

    # Articulation
    env_mean = np.mean(envelope_arr)
    env_max = np.max(envelope_arr) if np.max(envelope_arr) > 0 else 1.0
    fill_ratio = env_mean / env_max
    articulation = float(np.clip(1.0 - fill_ratio, 0.0, 1.0))

    # Bounciness (simplified without full audio)
    # We can't recompute crest factor from downsampled envelope,
    # so we rely more heavily on swing
    bounciness = float(np.clip(swing_ratio * 0.85, 0.0, 1.0))

    return articulation, bounciness
