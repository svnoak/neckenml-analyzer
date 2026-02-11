def analyze_vocal_presence(audio_array, sample_rate=16000, model_dir=None, vocal_model=None, input_is_embeddings=False):
    """
    Detects if vocals are present using a pre-trained Essentia model,
    but with a 'Noise Gate' to prevent analyzing silence/background noise.

    Args:
        audio_array: Audio signal array OR MusiCNN embeddings (if input_is_embeddings=True)
        sample_rate: Sample rate of audio (default 16000) - ignored if input_is_embeddings=True
        model_dir: Directory containing model files (default ~/.neckenml/models)
        vocal_model: Optional pre-initialized TensorflowPredictMusiCNN model (to avoid warnings)
        input_is_embeddings: If True, audio_array is MusiCNN embeddings, not raw audio
    """
    import essentia.standard as es
    import essentia
    import numpy as np
    import os

    essentia.log.info.active = False
    essentia.log.warning.active = False

    # 1. NOISE GATE (The new safety check) - skip if we already have embeddings
    if not input_is_embeddings:
        # Calculate RMS (Root Mean Square) to get average volume
        rms = np.sqrt(np.mean(audio_array**2))

        # Threshold: 0.01 is roughly -40dB.
        # If the track is quieter than this on average, it's likely background noise
        # or a recording error, and definitely doesn't have a strong lead vocal.
        NOISE_GATE_THRESHOLD = 0.005

        if rms < NOISE_GATE_THRESHOLD:
            print(f"      [VOCAL] Signal too weak (RMS: {rms:.4f} < {NOISE_GATE_THRESHOLD}). Skipping model.")
            return {
                "is_instrumental": True,
                "confidence": 0.0,  # 0% chance of vocals
                "label": "instrumental"
            }

    # Get model directory
    if model_dir is None:
        model_dir = os.path.expanduser("~/.neckenml/models")

    MODEL_FILENAME = "voice_instrumental-musicnn-msd-1.pb"
    model_path = os.path.join(model_dir, MODEL_FILENAME)

    if not os.path.exists(model_path):
        print(f"[WARNING] VOCAL MODEL MISSING at: {model_path}")
        return {"is_instrumental": True, "confidence": 0.0, "label": "error"}

    # 2. RUN DEEP LEARNING MODEL
    # Now we know there is actual signal, we run the heavy model.
    try:
        # If we already have embeddings, use TensorflowPredict2D for classification head
        if input_is_embeddings:
            # Use provided model instance or create new one
            if vocal_model is None:
                vocal_model = es.TensorflowPredict2D(
                    graphFilename=model_path,
                    output="model/Softmax"
                )

            # audio_array is actually MusiCNN embeddings matrix
            predictions = vocal_model(audio_array)
        else:
            if vocal_model is None:
                vocal_model = es.TensorflowPredictMusiCNN(
                    graphFilename=model_path,
                    output="model/Softmax"
                )

            # The model expects normalized inputs usually, but since we are handling
            # the gating ourselves, we can pass the audio. However, for the *prediction*
            # specifically, it is often safer to normalize the specific chunk going into
            # the Neural Net to match how it was trained, even if we preserve dynamics elsewhere.

            # Create a normalized copy just for this specific predictor
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                input_audio = audio_array / max_val
            else:
                input_audio = audio_array

            predictions = vocal_model(input_audio)

        # Predictions is usually [instrumental_prob, vocal_prob]
        # We average over the time frames to get one score for the file
        avg_preds = np.mean(predictions, axis=0)

        # Index 0 is usually instrumental, Index 1 is vocal for this specific model
        instrumental_score = avg_preds[0]
        vocal_score = avg_preds[1]

        is_instrumental = instrumental_score > vocal_score

        return {
            "is_instrumental": bool(is_instrumental),
            "confidence": float(vocal_score), # How confident are we it has vocals?
            "label": "instrumental" if is_instrumental else "voice",
            # Raw artifacts for persistence
            "raw_artifacts": {
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                "instrumental_score": float(instrumental_score),
                "vocal_score": float(vocal_score)
            }
        }

    except Exception as e:
        print(f"[VOCAL] Model failed: {e}. Defaulting to instrumental.")
        # Fallback to safe default
        return {
            "is_instrumental": True,
            "confidence": 0.0,
            "label": "instrumental"
        }
