METER_HINTS = {
    # Ternary (3/4)
    "hambo": 3, "hamburska": 3, "polska": 3, "pols": 3, 
    "springlek": 3, "bondpolska": 3, "slängpolska": 3, 
    "släng": 3, "vals": 3, "waltz": 3, "hoppvals": 3, 
    "mazurka": 3, "masurka": 3,
    # Binary (2/4 or 4/4)
    "schottis": [2, 4], "reinländer": [2, 4], "snoa": [2, 4], 
    "gånglåt": [2, 4], "marsch": [2, 4], "polka": [2, 4]
}

# =========================================================
# POLSKA vs HAMBO SIGNATURE PROFILES
# =========================================================
# Polska: The characteristic "lift" - beat 2 or 3 is elongated
# creating an asymmetric, "hanging" feel. High micro-timing variance.
# 
# Hambo: Heavy downbeat (beat 1 is longest), square phrasing,
# more predictable/regular timing.
# =========================================================

TERNARY_PROFILES = {
    # Profile: (ratio1_range, ratio2_range, ratio3_range, timing_variance_threshold)
    # Swedish Polska often has beat 3 elongated (the "lift")
    "polska_beat3_lift": {
        "ratios": (0.28, 0.38, 0.28, 0.38, 0.30, 0.45),  # r1_min, r1_max, r2_min, r2_max, r3_min, r3_max
        "variance_min": 0.003,  # Polska has higher micro-timing variance
        "description": "Beat 3 elongated (hanging feel)"
    },
    # Some Polska variants have beat 2 elongated
    "polska_beat2_lift": {
        "ratios": (0.28, 0.36, 0.35, 0.45, 0.25, 0.35),
        "variance_min": 0.003,
        "description": "Beat 2 elongated"
    },
    # Hambo: Heavy first beat
    "hambo_classic": {
        "ratios": (0.38, 0.50, 0.22, 0.35, 0.22, 0.35),
        "variance_max": 0.008,  # Hambo is more metronomic
        "description": "Heavy downbeat, square feel"
    },
    # Vals: Very even
    "vals_even": {
        "ratios": (0.30, 0.36, 0.30, 0.36, 0.30, 0.36),
        "variance_max": 0.005,
        "description": "Even triplet feel"
    }
}

class RhythmExtractor:
    def __init__(self):
        import numpy as np
        import madmom.features.downbeats
        import madmom.features.beats
        import essentia.standard as es

        # Use Madmom for beat tracking (better for rubato and folk music)
        self.beat_processor = madmom.features.beats.RNNBeatProcessor()
        self.downbeat_processor = madmom.features.downbeats.RNNDownBeatProcessor()

        # Keep Essentia as fallback
        self.rhythm_extractor_algo = es.RhythmExtractor2013(method="multifeature")

    def get_meter_hint(self, text: str):
        if not text: return None
        text = text.lower()
        for keyword, beats in METER_HINTS.items():
            if keyword in text: return beats
        return None

    def analyze_beats(self, file_path: str, metadata_context: str = "", return_artifacts: bool = False) -> tuple:
        """
        Extracts beat positions, confidence, and meter using Madmom RNN.

        Madmom is better for folk music because:
        - Handles rubato (tempo variations) in Polska
        - Provides beat activation functions (useful for detecting asymmetric patterns)
        - Better downbeat detection for dance music

        Args:
            file_path: Path to audio file
            metadata_context: Optional metadata for meter hints
            return_artifacts: If True, returns raw Madmom activation functions

        Returns:
            If return_artifacts=False: (audio, beats, beat_info, ternary_confidence)
            If return_artifacts=True: (audio, beats, beat_info, ternary_confidence, artifacts_dict)
        """
        import essentia.standard as es
        import numpy as np
        import madmom.features.beats

        # 1. LOAD AUDIO for both Madmom and folk feature extraction
        loader = es.MonoLoader(filename=file_path, sampleRate=44100)
        audio = loader()

        # Normalize for folk feature extraction (preserves dynamics for activations)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio_normalized = audio / max_val
        else:
            audio_normalized = audio

        # 2. MADMOM BEAT TRACKING
        # Process downbeats (gives us both beats and downbeats)
        activations = self.downbeat_processor(file_path)

        # Use DBNDownBeatTrackingProcessor to get beats from activations
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        beats_and_downbeats = tracker(activations)

        # beats_and_downbeats format: [[time, beat_number], ...]
        # beat_number: 1 = downbeat, 2/3/4 = other beats

        # 3. EXTRACT BEATS AND METER
        if len(beats_and_downbeats) == 0:
            # Fallback to simple beat tracking if downbeat detection fails
            beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            beats = beat_tracker(self.beat_processor(file_path))

            # Create dummy beat_info (no downbeat information)
            beat_info = [[t, 1] for t in beats]
            beats_per_bar = 4
            ternary_confidence = 0.5
        else:
            beats = beats_and_downbeats[:, 0]  # Extract just the times
            beat_positions = beats_and_downbeats[:, 1]  # Extract beat numbers

            # Detect meter from beat positions
            # Count how many beats per bar (max beat number)
            beats_per_bar = int(np.max(beat_positions)) if len(beat_positions) > 0 else 4

            # Ternary confidence based on detected meter
            ternary_confidence = 0.9 if beats_per_bar == 3 else 0.2

            # Build beat_info
            beat_info = beats_and_downbeats.tolist()

        # 4. Calculate BPM
        if len(beats) > 1:
            intervals = np.diff(beats)
            median_interval = np.median(intervals)
            bpm = 60.0 / median_interval if median_interval > 0 else 120.0
        else:
            bpm = 120.0

        # Convert beat_info to numpy array
        beat_info = np.array(beat_info)

        if return_artifacts:
            # Store ALL raw Madmom outputs - this is the key data!
            artifacts = {
                "source": "madmom_rnn_downbeat",
                "bpm": float(bpm),
                "beats": beats.tolist() if hasattr(beats, 'tolist') else list(beats),
                "beat_positions": beat_positions.tolist() if hasattr(beat_positions, 'tolist') else [],
                "activation_functions": activations.tolist() if hasattr(activations, 'tolist') else [],  # Raw RNN output!
                "beats_per_bar": int(beats_per_bar),
                "ternary_confidence": float(ternary_confidence),
                "fps": 100  # Activations frame rate
            }
            return audio_normalized, beats, beat_info, ternary_confidence, artifacts

        return audio_normalized, beats, beat_info, ternary_confidence
    
    def _calculate_ternary_confidence(self, beat_times):
        """
        Analyzes the activation pattern to determine if the track is likely ternary (3/4).
        """
        import numpy as np

        intervals = np.diff(beat_times)
        if len(intervals) < 6: return 0.5
        
        # Strategy 1: Grouping Variance (3s vs 2s vs 4s)
        def get_group_variance(n):
            groups = [sum(intervals[i:i+n]) for i in range(0, len(intervals)-n+1, n)]
            return np.var(groups) if len(groups) > 2 else float('inf')

        ternary_var = get_group_variance(3)
        binary_var = get_group_variance(2)
        quaternary_var = get_group_variance(4)
        
        # Strategy 2: Check for characteristic polska ratio patterns
        ratio_asymmetry = []
        for i in range(0, len(intervals) - 2, 3):
            total = sum(intervals[i:i+3])
            if total > 0:
                ratios = intervals[i:i+3] / total
                # How far from even (0.33, 0.33, 0.33)?
                ratio_asymmetry.append(sum(abs(r - 0.333) for r in ratios))
        
        avg_asymmetry = np.mean(ratio_asymmetry) if ratio_asymmetry else 0
        
        # Calculate final score
        conf = 0.5  # Neutral start
        
        min_var = min(ternary_var, binary_var, quaternary_var)
        
        # If 3-grouping is the most consistent
        if min_var == ternary_var:
            # Strong signal if it's much better than binary
            if ternary_var < binary_var * 0.7: conf += 0.3
            else: conf += 0.15
        elif min_var == binary_var:
             # If binary is extremely tight, it's definitely not ternary
            if binary_var < ternary_var * 0.5: conf -= 0.2
        
        # Polska asymmetry bonus (Polskas are ternary but often have high variance)
        if avg_asymmetry > 0.15: 
            conf += min(0.25, avg_asymmetry * 0.8)
        
        return np.clip(conf, 0.0, 1.0)

    def extract_folk_features(self, beat_times, audio_signal, return_artifacts=False):
        """
        Calculates specific features relevant to folk dance styles.

        Args:
            beat_times: Beat positions in seconds
            audio_signal: Audio waveform
            return_artifacts: If True, returns (features_dict, artifacts_dict)
        """
        import numpy as np

        if len(beat_times) < 12:
            if return_artifacts:
                return np.zeros(8), {}
            return np.zeros(8)

        # 1. Intervals & BPM
        ibis = np.diff(beat_times)
        avg_ibi = np.mean(ibis)
        if avg_ibi == 0: return np.zeros(8)
        bpm = 60.0 / avg_ibi

        bpm_stability = 1.0 - (np.std(ibis) / avg_ibi) # 1.0 is perfect, 0.0 is chaotic

        # 2. Ratios & Variances
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

        # 3. Energy/Activations (Volume Agnostic via tanh)
        beat_energy = 0.0
        activations = [] 
        window = int(44100 * 0.05)
        
        # We assume audio_signal is already normalized by analyze_beats logic
        # if it came from the return value of that function.
        for t in beat_times:
            idx = int(t * 44100)
            start, end = max(0, idx - window), min(len(audio_signal), idx + window)
            segment = audio_signal[start:end]
            energy = np.sum(segment**2) if len(segment) > 0 else 0
            beat_energy += energy
            activations.append(energy)
            
        punchiness = np.tanh((beat_energy / len(beat_times)) * 10) if len(beat_times) > 0 else 0

        # 4. Advanced Scoring
        polska_score, hambo_score = self._calculate_ternary_signatures(
            ratios=[r1_mean, r2_mean, r3_mean],
            triplet_variances=triplet_variances,
            intervals=ibis,
            activations=activations
        )

        features = {
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

        if return_artifacts:
            artifacts = {
                "beat_activations": [float(a) for a in activations],
                "intervals": ibis.tolist() if hasattr(ibis, 'tolist') else list(ibis),
                "triplet_variances": triplet_variances
            }
            return features, artifacts

        return features
    
    def _calculate_ternary_signatures(self, ratios, triplet_variances, intervals, activations):
        import numpy as np
        r1, r2, r3 = ratios
        
        timing_variance = np.mean(triplet_variances) if triplet_variances else 0.0
        interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0.0
        
        # Activation Analysis
        downbeat_dominance = 0.33
        if len(activations) >= 6:
            # Mean energy of Beat 1s
            avg_b1 = np.mean(activations[0::3])
            total = np.mean(activations) * 3
            if total > 0: downbeat_dominance = avg_b1 / total

        # --- SCORES ---
        polska_score = 0.0
        hambo_score = 0.0
        
        # Polska: Lift (Long 2 or 3) + Rubato (High Var) + Weak Downbeat
        if (r3 > r1 and r3 > 0.34) or (r2 > r1 and r2 > 0.36): polska_score += 0.35
        if timing_variance > 0.003: polska_score += min(0.25, timing_variance * 30)
        if downbeat_dominance < 0.38: polska_score += 0.15
        
        # Hambo: Heavy Downbeat + Strict (Low Var) + Strong Downbeat
        if r1 > 0.38: hambo_score += 0.30 + min(0.20, (r1 - 0.38) * 2)
        if timing_variance < 0.004: hambo_score += 0.20
        if downbeat_dominance > 0.40: hambo_score += 0.20
        
        return min(1.0, polska_score), min(1.0, hambo_score)

    def get_bars(self, beat_info):
        if len(beat_info) == 0: return []
        return [row[0] for row in beat_info if row[1] == 1.0]