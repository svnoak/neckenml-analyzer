

class StructureExtractor:
    def __init__(self):
        import essentia.standard as es
        self.w = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.eqloudness = es.EqualLoudness()
        self.mfcc = es.MFCC()
        self.sbic = es.SBic(cpw=1.5)

    def extract_segments(self, audio, bars: list[float], style_hint: str = None):
        """
        Tries Audio Analysis. If that fails or returns 0 sections, falls back to Math.
        """
        style = style_hint.lower() if style_hint else ""
        
        # 1. PREFER MATH FOR SQUARE STYLES
        # These styles are almost always 8/16 bars. Math is cleaner.
        if any(x in style for x in ['hambo', 'schottis', 'snoa', 'polka', 'vals', 'engelska']):
            return self._extract_segments_math(bars)

        # 2. TRY AUDIO ANALYSIS (For Polska/Unknown)
        segments = self._extract_segments_audio(audio, bars)

        # 3. FAILSAFE: If Audio found nothing (just [0.0]), Force Math
        if len(segments) < 2 and len(bars) > 4:
            print(f"[ERROR] Audio structure failed. Falling back to Grid Math.")
            return self._extract_segments_math(bars)
            
        return segments

    def _extract_segments_math(self, bars):
        """
        Calculates sections based on bar count (8, 12, 16, 32).
        """
        import numpy as np

        if not bars or len(bars) < 4: return [0.0]
        total_bars = len(bars)
        
        candidates = [8, 12, 16, 32]
        ratios = [abs((total_bars / c) - round(total_bars / c)) for c in candidates]
        phrase_bars = candidates[int(np.argmin(ratios))]

        # Heuristic: If long track, prefer 16 over 8
        if total_bars > 64 and phrase_bars < 16:
            phrase_bars = 16

        section_indices = list(range(0, total_bars, phrase_bars))
        sections = [bars[idx] for idx in section_indices]
        sections = sorted(list(set(sections)))
        
        if len(sections) == 0: return [0.0]
        return [float(s) for s in sections]

    def _extract_segments_audio(self, audio, bars):
        """
        Uses Essentia SBic to find texture changes.
        """
        import essentia.standard as es
        import numpy as np

        mfccs = []
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024, startFromZero=True):
            spec = self.spectrum(self.w(frame))
            eq_spec = self.eqloudness(spec)
            _, mfcc_coeffs = self.mfcc(eq_spec)
            mfccs.append(mfcc_coeffs)

        mfccs = np.array(mfccs)
        if len(mfccs) < 200: return [0.0]

        try:
            segments = self.sbic(mfccs)
            raw_times = [s * 1024.0 / 16000.0 for s in segments]

            # Snap to grid
            snapped_sections = [0.0]
            if bars and len(bars) > 0:
                for t in raw_times:
                    closest_bar = min(bars, key=lambda x: abs(x - t))
                    # Min duration check (8s)
                    if (closest_bar - snapped_sections[-1]) > 8.0:
                        snapped_sections.append(closest_bar)
            else:
                snapped_sections = [0.0] + raw_times

            return [float(s) for s in snapped_sections]

        except Exception as e:
            print(f"[ERROR] SBic failed: {e}")
            return [0.0]