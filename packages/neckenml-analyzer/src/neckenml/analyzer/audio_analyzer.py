import traceback
import os
import gc
from typing import Optional

from .extractors.vocal import analyze_vocal_presence
from .extractors.swing import calculate_swing_ratio
from .extractors.feel import analyze_feel
from .extractors.section_labeler import ABSectionLabeler
from neckenml.core.sources.base import AudioSource

class AudioAnalyzer:
    def __init__(self, audio_source: Optional[AudioSource] = None, model_dir: Optional[str] = None):
        """
        Initialize AudioAnalyzer.

        Args:
            audio_source: AudioSource implementation for fetching audio files. Optional.
            model_dir: Directory containing MusiCNN model files. If None, uses ~/.neckenml/models
        """
        print("Loading Analysis Models...")

        # Delayed imports to avoid circular dependency issues during startup
        from .extractors.rhythm import RhythmExtractor
        from .extractors.structure import StructureExtractor
        from neckenml.core.classifier.style_head import ClassificationHead
        from neckenml.core.folk_authenticity import FolkAuthenticityDetector
        import essentia.standard as es

        self.audio_source = audio_source
        self.model_dir = self._get_model_dir(model_dir)

        self.rhythm_extractor = RhythmExtractor()
        self.structure_extractor = StructureExtractor()
        self.head = ClassificationHead()
        self.folk_detector = FolkAuthenticityDetector(manual_review_threshold=0.6)

        self.loudness_algo = es.LoudnessEBUR128()
        self.rms_algo = es.RMS()
        self.zcr_algo = es.ZeroCrossingRate()
        self.onset_rate_algo = es.OnsetRate()
        self.envelope_algo = es.Envelope(attackTime=10, releaseTime=50)

        self.tf_embeddings = None
        self.vocal_model = None
        self._load_ai_models()

    def __enter__(self):
        """Enables use in a 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically called when exiting the 'with' block."""
        self.close()

    def close(self):
        """
        Manually frees memory and destroys the internal TensorFlow graph.
        Call this when you are done with the analyzer to prevent memory leaks.
        """
        print("   [CLEANUP] NeckenML: Releasing analyzer resources...")

        # 1. Delete TensorFlow model references
        if self.tf_embeddings:
            del self.tf_embeddings
            self.tf_embeddings = None

        if self.vocal_model:
            del self.vocal_model
            self.vocal_model = None

        # 2. Delete Essentia algorithm instances
        if hasattr(self, 'loudness_algo'):
            del self.loudness_algo
        if hasattr(self, 'rms_algo'):
            del self.rms_algo
        if hasattr(self, 'zcr_algo'):
            del self.zcr_algo
        if hasattr(self, 'onset_rate_algo'):
            del self.onset_rate_algo
        if hasattr(self, 'envelope_algo'):
            del self.envelope_algo

        # 3. Delete heavy extractor objects
        if hasattr(self, 'rhythm_extractor'):
            del self.rhythm_extractor
        if hasattr(self, 'structure_extractor'):
            del self.structure_extractor
        if hasattr(self, 'head'):
            del self.head
        if hasattr(self, 'folk_detector'):
            del self.folk_detector

        # 5. Force aggressive garbage collection (multiple passes for circular refs)
        gc.collect()
        gc.collect()
        gc.collect()
        print("   [CLEANUP] NeckenML: Resources freed.")

    def _get_model_dir(self, model_dir: Optional[str]) -> str:
        """Get the model directory, using default if not provided."""
        if model_dir:
            return os.path.expanduser(model_dir)

        default_dir = os.path.expanduser("~/.neckenml/models")
        if not os.path.exists(default_dir):
            print(f"Default model directory doesn't exist: {default_dir}")
            print(f"Create it and download models from https://essentia.upf.edu/models/")
        return default_dir

    def _load_ai_models(self):
        import essentia.standard as es
        try:
            model_backbone = os.path.join(self.model_dir, "msd-musicnn-1.pb")
            if not os.path.exists(model_backbone):
                # Don't crash immediately, allow retry, but log error
                print(f"    Model file missing: {model_backbone}")
                self.tf_embeddings = None
                return

            self.tf_embeddings = es.TensorflowPredictMusiCNN(
                graphFilename=model_backbone,
                output="model/dense/BiasAdd"
            )

            # Load vocal model (uses TensorflowPredictMusiCNN for raw audio input)
            vocal_model_path = os.path.join(self.model_dir, "voice_instrumental-musicnn-msd-1.pb")
            if os.path.exists(vocal_model_path):
                self.vocal_model = es.TensorflowPredictMusiCNN(
                    graphFilename=vocal_model_path,
                    output="model/Sigmoid"  # Using Sigmoid activation for binary classification
                )
            else:
                print(f"    Vocal model file missing: {vocal_model_path}")
                self.vocal_model = None

            print("Pipeline loaded.")
        except Exception as e:
            print(f"Models failed to load: {e}")
            self.tf_embeddings = None
            self.vocal_model = None

    def analyze(self, track_id: str, metadata_context: str = "") -> dict:
        """
        Analyze a track by fetching audio from the audio_source.

        Args:
            track_id: Track identifier to pass to audio_source
            metadata_context: Optional metadata context for analysis

        Returns:
            dict: Analysis results

        Raises:
            ValueError: If no audio_source was provided
        """
        if self.audio_source is None:
            raise ValueError("No audio_source provided. Use analyze_file() for direct file analysis.")

        # Fetch audio file from source
        file_path = self.audio_source.fetch_audio(track_id)

        try:
            # Analyze the file
            result = self.analyze_file(file_path, metadata_context)
            return result
        finally:
            # Clean up temporary files if needed
            self.audio_source.cleanup(file_path)

    def _extract_lightweight_features(self, audio) -> list:
        """
        Extracts robust, crash-proof structural proxies.
        Returns: [RMS (Dynamics), ZCR (Texture), OnsetRate (Busyness)]
        """
        try:
            # 1. Dynamics (Standard Deviation of RMS amplitude)
            rms = self.rms_algo(audio)

            # 2. Texture (Zero Crossing Rate)
            zcr = self.zcr_algo(audio)

            # 3. Busyness (Onset Rate)
            # This is valuable for distinguishing "busy" Polskas from "smooth" Waltzes
            # OnsetRate returns [onset_rate, num_onsets]
            onset_rate, _ = self.onset_rate_algo(audio)

            return [float(rms), float(zcr), float(onset_rate)]
        except Exception:
            # Fallback (return 3 zeros since we expect 3 features now)
            return [0.0, 0.0, 0.0]

    def analyze_file(self, file_path: str, metadata_context: str = "", return_artifacts: bool = False) -> dict:
        import essentia.standard as es
        import numpy as np
        try:
            # --- 1. LOAD AUDIO ---
            print(f"   [ANALYSIS] Load audio and texture...")
            loader = es.MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)
            audio_16k = loader()

            # --- ROBUST LOUDNESS CALCULATION ---
            # This handles the "length-1 array" crash by force-flattening the result.
            # Using reusable algorithm instance to avoid "No network created" warning
            loudness = -14.0 # Default
            try:
                stereo_audio = np.stack([audio_16k, audio_16k], axis=1)
                loudness_results = self.loudness_algo(stereo_audio)

                # Unwrap: Essentia might return a scalar, a 1D array, or a tuple of arrays.
                # We want the first value (Integrated Loudness).
                first_val = loudness_results[0]

                # Convert to numpy array to unify handling, then flatten and grab index 0.
                # This works for scalars (becomes [x]), 1D arrays ([x, x]), etc.
                loudness = float(np.array(first_val).flatten()[0])
            except Exception as e:
                print(f"   Loudness calc warning: {e}")

            # --- EMBEDDINGS ---
            print(f"   [ANALYSIS] Generating embeddings...")
            if self.tf_embeddings is None:
                raise RuntimeError("MusiCNN embeddings model not loaded")
            
            # Safe Normalization for Neural Net
            max_val = np.max(np.abs(audio_16k))
            audio_for_nn = (audio_16k / max_val) if max_val > 0 else audio_16k

            raw_embeddings = self.tf_embeddings(audio_for_nn)
            avg_embedding = np.mean(raw_embeddings, axis=0)

            # --- 2. VOCALS ---
            print(f"   [ANALYSIS] Doing voice detection...")
            # Vocal model expects raw audio, not embeddings
            vocal_data = analyze_vocal_presence(
                audio_for_nn,
                sample_rate=16000,
                model_dir=self.model_dir,
                vocal_model=self.vocal_model,
                input_is_embeddings=False  # Changed to False - model expects audio
            )

            # --- 3. RHYTHM ---
            print(f"   [ANALYSIS] Doing rhythm analysis...")
            if return_artifacts:
                act, beat_times, beat_info, ternary_confidence, beat_artifacts = self.rhythm_extractor.analyze_beats(
                    file_path, metadata_context, return_artifacts=True
                )
                folk_features, rhythm_artifacts = self.rhythm_extractor.extract_folk_features(beat_times, act, return_artifacts=True)
            else:
                act, beat_times, beat_info, ternary_confidence = self.rhythm_extractor.analyze_beats(file_path, metadata_context)
                folk_features = self.rhythm_extractor.extract_folk_features(beat_times, act)
                beat_artifacts = {}
                rhythm_artifacts = {}

            bars = self.rhythm_extractor.get_bars(beat_info)

            # --- 4. FEEL ---
            print(f"   [ANALYSIS] Doing swing & feel analysis...")
            if return_artifacts:
                swing_ratio, swing_artifacts = calculate_swing_ratio(file_path, beat_times, return_artifacts=True)
                feel_data, feel_artifacts = analyze_feel(audio_16k, beat_times, swing_ratio, self.envelope_algo, return_artifacts=True)
            else:
                swing_ratio = calculate_swing_ratio(file_path, beat_times)
                feel_data = analyze_feel(audio_16k, beat_times, swing_ratio, self.envelope_algo)
                swing_artifacts = {}
                feel_artifacts = {}

            # --- 5. STATS ---
            print(f"   [ANALYSIS] Extracting layout stats...")
            layout_stats = self._extract_lightweight_features(audio_16k)

            voice_conf = float(vocal_data['confidence'])
            articulation = float(feel_data['articulation'])
            bounciness = float(feel_data['bounciness'])
            ternary_conf = float(ternary_confidence)

            # --- 6. PREDICT ---
            print(f"   [ANALYSIS] Predict style...")
            # Note: Ensure ClassificationHead.EXPECTED_FEATURE_COUNT = 217
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
                [voice_conf],       # 1
                [articulation],     # 1
                [bounciness]        # 1
            ])
            
            predicted_style, ml_confidence = self.head.predict(full_vector)

            # --- 7. AUTHENTICITY ---
            print(f"   [ANALYSIS] Checking authenticity...")
            folk_auth_result = self.folk_detector.analyze(
                rms_value=layout_stats[0],
                zcr_value=layout_stats[1],
                swing_ratio=swing_ratio,
                articulation=articulation,
                bounciness=bounciness,
                voice_probability=voice_conf,
                is_likely_instrumental=vocal_data['is_instrumental'],
                embedding=avg_embedding
            )

            # --- 8. STRUCTURE --- 
            print(f"   [ANALYSIS] Structure analysis (Hint: {predicted_style})...")
            try:
                sections = self.structure_extractor.extract_segments(
                    audio=audio_16k, 
                    bars=bars, 
                    style_hint=predicted_style
                )
                if sections is None: sections = []
            except Exception as e:
                print(f"    Structure analysis unstable: {e}")
                sections = [] 

            if sections:
                section_labeler = ABSectionLabeler(sr=16000)
                section_labels = section_labeler.label_sections(audio_16k, sections)
            else:
                section_labels = []

            # --- 9. RETURN ---
            print(f"   [ANALYSIS] Returning metrics...")
            punchiness = folk_features["punchiness"]
            polska_score = folk_features.get('polska_score', 0.0)
            hambo_score = folk_features.get('hambo_score', 0.0)
            bpm_stability = folk_features.get('bpm_stability', 0.0)
            meter_numerator = int(np.max(beat_info[:,1])) if len(beat_info) > 0 else 0

            raw_result = {
                "ml_suggested_style": predicted_style,
                "ml_confidence": float(ml_confidence),
                "embedding": full_vector.tolist(),
                "loudness_lufs": loudness,
                "tempo_bpm": folk_features["bpm"],
                "bpm_stability": bpm_stability,
                "is_likely_instrumental": bool(vocal_data['is_instrumental']),
                "voice_probability": voice_conf,
                "swing_ratio": float(swing_ratio),
                "articulation": articulation,
                "bounciness": bounciness,
                "avg_beat_ratios": [
                    folk_features["r1_mean"],
                    folk_features["r2_mean"],
                    folk_features["r3_mean"]
                ],
                "punchiness": punchiness,
                "polska_score": polska_score,
                "hambo_score": hambo_score,
                "ternary_confidence": ternary_conf,
                "meter": f"{meter_numerator}/4",
                "bars": [float(b) for b in bars],
                "beat_times": [float(b) for b in beat_times],  # Store for recalculation
                "sections": [float(s) for s in sections],
                "section_labels": section_labels,
                "folk_authenticity_score": float(folk_auth_result['folk_authenticity_score']),
                "requires_manual_review": bool(folk_auth_result['requires_manual_review']),
                "folk_authenticity_breakdown": folk_auth_result['confidence_breakdown'],
                "folk_authenticity_interpretation": folk_auth_result['interpretation']
            }

            # If artifacts requested, build and return them separately
            if return_artifacts:
                # Get audio duration
                import librosa
                duration = librosa.get_duration(path=file_path)

                artifacts = {
                    "version": "1.0.0",
                    "rhythm_extractor": {
                        # Store ALL raw outputs from Essentia RhythmExtractor2013
                        **beat_artifacts,  # Contains: bpm, beats, beats_confidence, beats_intervals, estimates, beats_per_bar, ternary_confidence
                        "beat_info": beat_info.tolist() if hasattr(beat_info, 'tolist') else beat_info,
                        "bars": [float(b) for b in bars]
                    },
                    "musicnn": {
                        "raw_embeddings": raw_embeddings.tolist() if hasattr(raw_embeddings, 'tolist') else raw_embeddings,
                        "avg_embedding": avg_embedding.tolist() if hasattr(avg_embedding, 'tolist') else avg_embedding
                    },
                    "vocal": vocal_data.get('raw_artifacts', {
                        "instrumental_score": float(vocal_data.get('confidence', 0.0)),
                        "vocal_score": 1.0 - float(vocal_data.get('confidence', 0.0))
                    }),
                    "audio_stats": {
                        "loudness_lufs": float(loudness),
                        "rms": float(layout_stats[0]),
                        "zcr": float(layout_stats[1]),
                        "onset_rate": float(layout_stats[2]),
                        "duration_seconds": float(duration)
                    },
                    "onsets": swing_artifacts,
                    "structure": {
                        "sections": [float(s) for s in sections],
                        "section_labels": section_labels
                    },
                    "dynamics": {
                        **feel_artifacts,
                        **rhythm_artifacts
                    }
                }

                result_with_artifacts = {
                    "features": self._sanitize_for_json(raw_result),
                    "raw_artifacts": self._sanitize_for_json(artifacts)
                }

                # Clean up large numpy arrays immediately
                del audio_16k, audio_for_nn, raw_embeddings, avg_embedding
                if 'stereo_audio' in locals():
                    del stereo_audio
                gc.collect()

                return result_with_artifacts

            # Clean up large numpy arrays immediately
            del audio_16k, audio_for_nn, raw_embeddings, avg_embedding
            if 'stereo_audio' in locals():
                del stereo_audio
            gc.collect()

            return self._sanitize_for_json(raw_result)

        except Exception as e:
            print(f"Analysis Error: {e}")
            traceback.print_exc()
            return None

    def refine_structure(self, file_path: str, bars: list, new_style_hint: str, audio_array=None) -> dict:
        import essentia.standard as es
        try:
            print(f"   [ANALYSIS] Refining structure with hint: {new_style_hint}...")
            
            if audio_array is not None:
                audio_16k = audio_array
            else:
                loader = es.MonoLoader(filename=file_path, sampleRate=16000)
                audio_16k = loader()

            sections = self.structure_extractor.extract_segments(
                audio=audio_16k, 
                bars=bars, 
                style_hint=new_style_hint
            )

            if sections:
                section_labeler = ABSectionLabeler(sr=16000)
                labels = section_labeler.label_sections(audio_16k, sections)
            else:
                labels = []

            result = {
                "sections": [self._to_float(s) for s in sections],
                "section_labels": labels
            }
            return self._sanitize_for_json(result)
        except Exception as e:
            print(f"Structure Refinement Error: {e}")
            return None

    def _to_float(self, value):
        try:
            if hasattr(value, 'item'): return float(value.item())
            if isinstance(value, (list, tuple)): return float(value[0]) if value else 0.0
            return float(value)
        except: return 0.0

    def _sanitize_for_json(self, data):
        import numpy as np
        """Recursively converts numpy types to native Python types."""
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_json(i) for i in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.bool_, bool)):
            return bool(data)
        else:
            return data