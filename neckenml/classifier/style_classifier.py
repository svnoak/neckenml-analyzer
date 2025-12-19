from typing import Optional, Tuple, Callable, Any


class StyleClassifier:
    """
    Multi-Label Classifier for Swedish Folk Music.

    Decision Priority:
    1. Metadata (Keywords in Title/Album) -> 98% Confidence
    2. AI Brain (Texture + Rhythmic Fingerprint) -> 85% Confidence
    3. Heuristics (Math on BPM, Swing, Ratios, Structure) -> 40-75% Confidence

    Keywords are now loaded from the database (style_keywords table) via a cache.
    This allows admin updates without code redeployment.
    """

    def __init__(
        self,
        db: Any = None,
        categorize_tempo_fn: Optional[Callable] = None,
        get_keywords_fn: Optional[Callable] = None
    ):
        """
        Initialize StyleClassifier.

        Args:
            db: Database session (optional, for keyword lookups)
            categorize_tempo_fn: Function to categorize tempo (style, effective_bpm) -> tempo_category
            get_keywords_fn: Function to get keywords from database (db) -> list of (keyword, main_style, sub_style)
        """
        from neckenml.classifier.style_head import ClassificationHead
        self.head = ClassificationHead()

        self._db = db
        self._categorize_tempo = categorize_tempo_fn
        self._get_keywords = get_keywords_fn

    def classify(self, track: Any, analysis: dict) -> list:
        """
        Returns a list of classifications (Primary + Secondaries).
        """
        results = []
        raw_bpm = analysis.get("tempo_bpm", 0)
        
        # --- A. DETECT PRIMARY STYLE ---
        primary = self._get_primary_style(track, analysis)
        
        # Calculate Multiplier & Effective BPM (Normalization)
        primary['multiplier'], primary['effective_bpm'] = self._calculate_mpm(primary['style'], raw_bpm)
        if self._categorize_tempo:
            primary['dance_tempo'] = self._categorize_tempo(primary['style'], primary['effective_bpm'])
        else:
            primary['dance_tempo'] = None
        primary['type'] = 'Primary'
        
        results.append(primary)

        # --- B. ADD COMPATIBLE STYLES ---
        secondaries = self._get_secondary_styles(primary, raw_bpm, analysis)
        results.extend(secondaries)

        return results

    # ====================================================
    # 1. CORE LOGIC
    # ====================================================

    def _get_primary_style(self, track, analysis):
        """
        Decides the main style.
        Now returns sub_style when matched via metadata.
        """

        import numpy as np

        # --- 1. GOD RULE: Metadata ---
        # If the artist calls it a Hambo, it is a Hambo.
        print(f"Checking metadata for: {track.title}")
        main_style, sub_style = self._check_metadata(track)
        if main_style:
            style_desc = main_style + (f"/{sub_style}" if sub_style else "")
            print(f"Using metadata classification: {style_desc}")
            return {
                "style": main_style,
                "sub_style": sub_style,
                "confidence": 0.98,
                "reason": f"Metadata match: '{style_desc}'",
                "source": "metadata"
            }
        
        # --- 2. ML Suggestion ---
        # Check if the neural network recognizes the texture/rhythm fingerprint
        embedding = analysis.get("embedding")
        if embedding:
            ml_style, ml_confidence = self.head.predict(embedding)
            if ml_style != "Unknown":
                return {
                    "style": ml_style,
                    "confidence": ml_confidence,  # Real probability from model
                    "reason": "AI Groove Fingerprint",
                    "source": "ml"
                }

        # --- 3. FALLBACK: Heuristics (Math + Structure) ---
        # These are educated guesses - confidence reflects uncertainty
        meter = analysis.get("meter", "4/4")
        swing = analysis.get("swing_ratio", 1.0)
        bpm = analysis.get("tempo_bpm", 0)
        ratios = analysis.get("avg_beat_ratios", [0.33, 0.33, 0.33])
        punchiness = analysis.get("punchiness", 0)
        
        # Polska/Hambo signature scores from rhythm analysis
        polska_score = analysis.get("polska_score", 0.0)
        hambo_score = analysis.get("hambo_score", 0.0)
        
        # Ternary confidence helps detect Polska misidentified as binary
        ternary_confidence = analysis.get("ternary_confidence", 0.5)
        
        # Structural Analysis
        bars = analysis.get("bars", [])
        sections = analysis.get("sections", [])
        
        # Calculate Phrase Length
        avg_bars = 0
        if len(sections) > 1 and len(bars) > 0:
            lengths = []
            for i in range(len(sections) - 1):
                start = sections[i]
                end = sections[i+1]
                num_bars = len([b for b in bars if start <= b < end])
                if num_bars > 4: 
                    lengths.append(num_bars)
            if lengths:
                avg_bars = np.median(lengths)

        # Helper: Is it "Square" (8, 16, 32)?
        def is_square(val):
            if 7.0 <= val <= 9.0: return True   # 8 bars
            if 15.0 <= val <= 17.0: return True # 16 bars (A+A)
            if 30.0 <= val <= 34.0: return True # 32 bars
            return False
        
        # Helper: Build heuristic result with honest confidence
        # Heuristics max out at 0.50 to indicate they're guesses
        def heuristic_result(style, base_confidence, reason, bonus=0.0):
            # Cap heuristic confidence at 0.50 (50%) - these are guesses
            conf = min(0.50, base_confidence + bonus)
            return {
                "style": style, 
                "confidence": conf, 
                "reason": f"[Heuristic] {reason}",
                "source": "heuristic"
            }

        # === TERNARY METER (3/4) ===
        if "3/" in meter:
            # Use Polska/Hambo signature scores first (more reliable)
            score_diff = polska_score - hambo_score
            
            # Strong Polska signature
            if polska_score > 0.45 and score_diff > 0.15:
                conf = min(0.50, 0.35 + polska_score * 0.20)
                reason = f"Polska signature ({polska_score:.2f}): "
                if ratios[2] > ratios[0]:
                    reason += "Beat 3 lift"
                elif ratios[1] > ratios[0]:
                    reason += "Beat 2 lift"
                else:
                    reason += "Asymmetric timing"
                return heuristic_result("Polska", conf, reason)
            
            # Strong Hambo signature
            if hambo_score > 0.45 and score_diff < -0.10:
                bonus = 0.10 if is_square(avg_bars) else 0.0
                conf = min(0.50, 0.35 + hambo_score * 0.20)
                reason = f"Hambo signature ({hambo_score:.2f}): Heavy downbeat"
                if is_square(avg_bars):
                    reason += f" + Square ({int(avg_bars)} bars)"
                return heuristic_result("Hambo", conf, reason, bonus)
            
            # Fallback to ratio-based logic when signatures are inconclusive
            # Hambo Logic: Long 1st beat + Square Structure
            if ratios[0] > 0.40:
                bonus = 0.10 if is_square(avg_bars) else 0.0
                reason = "Long 1st Beat" + (f" + Square ({int(avg_bars)} bars)" if is_square(avg_bars) else "")
                return heuristic_result("Hambo", 0.40, reason, bonus)
            
            # Vals Logic: Even beats
            if abs(ratios[0] - 0.33) < 0.05 and abs(ratios[1] - 0.33) < 0.05:
                return heuristic_result("Vals", 0.35, "Even Beat lengths")

            # Slängpolska Logic: Smooth
            if punchiness < 0.1: 
                return heuristic_result("Slängpolska", 0.35, "Smooth/Flowing Texture")

            # If we have ANY Polska signal, prefer it over Unknown
            if polska_score > 0.25:
                return heuristic_result("Polska", 0.30, f"Weak Polska signal ({polska_score:.2f})")

            # Polska Logic: Generic/Asymmetric (Often NOT square)
            return {
                "style": "Unknown", 
                "confidence": 0.0, 
                "reason": "Undetermined 3/4 Rhythm",
                "source": "heuristic"
            }

        # === BINARY METER (2/4 or 4/4) ===
        else:
            # Debug logging for polska/polka decisions
            print(f"Binary meter detected - checking for polska misdetection:")
            print(f"ternary_conf={ternary_confidence:.2f}, polska_score={polska_score:.2f}")
            print(f"bpm={bpm:.0f}, swing={swing:.2f}, ratios={[f'{r:.2f}' for r in ratios]}")
            
            # ============================================================
            # POLSKA RESCUE: Check if this might actually be a misdetected Polska
            # Essentia often struggles with asymmetric 3/4 polska rhythm
            # ============================================================
            if self._is_likely_misdetected_polska(
                ternary_confidence, polska_score, ratios, bpm, swing
            ):
                reason = f"Rescued from binary: ternary_conf={ternary_confidence:.2f}, polska_score={polska_score:.2f}"
                print(f"POLSKA RESCUE triggered!")
                return heuristic_result("Polska", 0.40, reason)
            
            # Schottis Logic: High Swing + Square Structure
            if swing > 1.25:
                bonus = 0.10 if is_square(avg_bars) else 0.0
                reason = f"High Swing ({swing:.2f})" + (" + Square" if is_square(avg_bars) else "")
                return heuristic_result("Schottis", 0.40, reason, bonus)
            
            # Snoa Logic: Strict Tempo Range
            if 80 < bpm < 115:
                return heuristic_result("Snoa", 0.35, f"Walking Tempo ({int(bpm)} BPM)")
            
            # Polka Logic: Fast Tempo
            # Additional check: Make sure it's not a polska in disguise
            elif bpm >= 115:
                # Only override to Polska if BOTH indicators are very strong
                # This prevents real Polka from being misclassified
                if ternary_confidence > 0.65 and polska_score > 0.45:
                    return heuristic_result("Polska", 0.35, 
                        f"Fast 3/4 (ternary={ternary_confidence:.2f}, polska={polska_score:.2f})")
                return heuristic_result("Polka", 0.30, f"Fast Tempo ({int(bpm)} BPM)")
            
            return {
                "style": "Unknown", 
                "confidence": 0.0, 
                "reason": "Undetermined Binary Rhythm",
                "source": "heuristic"
            }
    
    def _is_likely_misdetected_polska(self, ternary_conf, polska_score, ratios, bpm, swing):
        """
        Heuristic to detect if a track classified as binary is actually a Polska.
        
        BE CONSERVATIVE - only rescue clear misdetections, don't accidentally 
        convert real Polka to Polska!
        
        Key differences:
        - Polska: 3/4, asymmetric beats, 100-130 BPM (bar tempo), rubato timing
        - Polka: 2/4, even beats, 120-160 BPM, strict timing, swing ~1.0
        """
        signals = 0
        
        # REQUIRED: Must have meaningful ternary confidence
        # This is the primary indicator that meter detection was wrong
        if ternary_conf < 0.45:
            return False  # Not enough evidence of ternary meter
        
        if ternary_conf > 0.65:
            signals += 2
        elif ternary_conf > 0.55:
            signals += 1
        elif ternary_conf >= 0.50:
            signals += 1  # At threshold - still counts as a signal
        
        # Polska score requirement depends on ternary confidence
        # Higher ternary confidence = we can accept lower polska score
        min_polska_score = 0.25 if ternary_conf >= 0.55 else 0.15
        
        if polska_score < min_polska_score:
            return False  # No polska rhythmic characteristics
            
        if polska_score > 0.50:
            signals += 2
        elif polska_score > 0.35:
            signals += 1
        elif polska_score > 0.20:
            signals += 1  # Weak but present polska signal
        
        # Tempo check: Polska bar tempo is typically 100-135
        # Real polka is typically 120-160 BPM in 2/4
        # BPM > 115 is more likely polka territory
        if bpm > 115:
            signals -= 1  # Penalty for polka-typical tempo
        elif 95 < bpm <= 115:
            signals += 1  # Polska-typical tempo range
        
        # Asymmetric ratios are a strong polska indicator
        # Polka should have very even beats (close to 0.5, 0.5 or 0.25x4)
        ratio_spread = max(ratios) - min(ratios)
        if ratio_spread > 0.12:  # Strong asymmetry
            signals += 1
        
        # Swing close to 1.0 suggests strict mechanical timing (Polka)
        # Polska typically has more rubato/variation
        if 0.95 <= swing <= 1.05:
            signals -= 1  # Penalty for very even swing (polka-like)
        elif swing < 0.90 or swing > 1.10:
            signals += 1  # Rubato timing suggests polska
        
        # High swing actually suggests schottis, not polska
        if swing > 1.20:
            signals -= 1  # Penalty for high swing
        
        # Need at least 3 signals - be balanced between rescue and false positives
        return signals >= 3

    def _get_secondary_styles(self, primary, raw_bpm, analysis):
        """
        Determines compatible secondary styles.
        """
        style = primary['style']
        base_conf = primary['confidence']
        swing = analysis.get("swing_ratio", 1.0)
        secondaries = []

        def add_secondary(new_style, reason, confidence_penalty=0.8):
            mult, eff = self._calculate_mpm(new_style, raw_bpm)
            secondaries.append({
                "style": new_style,
                "type": "Secondary",
                "confidence": base_conf * confidence_penalty,
                "reason": reason,
                "multiplier": mult,
                "effective_bpm": eff,
                "dance_tempo": self._categorize_tempo(new_style, eff) if self._categorize_tempo else None
            })

        if style == "Snoa":
            add_secondary("Polka", "Compatible Rhythm (Slow Polka)")
        elif style == "Polka" and raw_bpm < 120:
            add_secondary("Snoa", "Compatible Rhythm (Fast Snoa)")

        if style == "Engelska":
            add_secondary("Polka", "Rhythmically Identical")

        if style == "Schottis" and swing < 1.4:
            add_secondary("Polka", "Low Swing Schottis", confidence_penalty=0.6)

        if style == "Vals":
            add_secondary("Polska", "Smooth 3/4", confidence_penalty=0.5)

        return secondaries

    # ====================================================
    # 4. HELPERS
    # ====================================================

    def _calculate_mpm(self, style: str, raw_bpm: float) -> tuple[float, int]:
        if not raw_bpm: return 1.0, 0
            
        multiplier = 1.0
        
        if style == "Hambo":
            if raw_bpm > 160: multiplier = 0.333
            elif raw_bpm < 70: multiplier = 2.0
        elif style in ["Polska", "Slängpolska"]:
            if raw_bpm > 180: multiplier = 0.5
        elif style == "Schottis":
            if raw_bpm > 200: multiplier = 0.5
            elif raw_bpm < 75: multiplier = 2.0
        elif style == "Vals":
            if raw_bpm > 100: multiplier = 0.333 

        effective_bpm = int(raw_bpm * multiplier)
        return multiplier, effective_bpm

    def _check_metadata(self, track) -> Tuple[Optional[str], Optional[str]]:
        """
        Check metadata for dance style keywords.
        Priority: Track title > Album title > Artist names
        Keywords are loaded from database and checked longest-first to avoid substring matches
        (e.g., "slängpolska" should match before "polska")

        Returns:
            Tuple of (main_style, sub_style) or (None, None) if no match.
        """
        if not self._get_keywords or not self._db:
            return (None, None)

        try:
            artist_names = []
            if hasattr(track, 'artist_links') and track.artist_links:
                artist_names = [link.artist.name for link in track.artist_links]

            album_title = track.album.title if hasattr(track, 'album') and track.album else ""
            track_title = track.title.lower() if hasattr(track, 'title') else ""

            print(f"Metadata check - Title: '{track_title}', Album: '{album_title}'")

            sorted_keywords = self._get_keywords(self._db)

            # 1. FIRST: Check track title (highest priority)
            for keyword, main_style, sub_style in sorted_keywords:
                if keyword in track_title:
                    style_desc = main_style + (f"/{sub_style}" if sub_style else "")
                    print(f"Title match: '{keyword}' -> {style_desc}")
                    return (main_style, sub_style)

            # 2. SECOND: Check album title (lower priority)
            album_lower = album_title.lower()
            for keyword, main_style, sub_style in sorted_keywords:
                if keyword in album_lower:
                    print(f"Album match: '{keyword}' -> {main_style}")
                    return (main_style, sub_style)

            # 3. THIRD: Check artist names (lowest priority, rarely useful)
            artist_text = ' '.join(artist_names).lower()
            for keyword, main_style, sub_style in sorted_keywords:
                if keyword in artist_text:
                    print(f"Artist match: '{keyword}' -> {main_style}")
                    return (main_style, sub_style)

            return (None, None)
        except Exception as e:
            print(f"   Metadata check error: {e}")
            return (None, None)