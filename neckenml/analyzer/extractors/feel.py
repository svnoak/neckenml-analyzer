def analyze_feel(audio, beat_times, swing_ratio):
    """
    Analyzes the texture of the audio (Staccato vs Legato, Bounciness)
    independent of the recording volume.
    """
    import numpy as np
    import essentia.standard as es

    if len(audio) == 0:
        return {'articulation': 0.0, 'bounciness': 0.0}

    # --- 1. LOCAL NORMALIZATION ---
    # We normalize just for this function because we are analyzing the SHAPE
    # of the waveform (envelope), not the absolute loudness.
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        norm_audio = audio / max_val
    else:
        return {'articulation': 0.0, 'bounciness': 0.0} # Silent track

    # --- 2. CALCULATE ENVELOPE ---
    # The envelope traces the "outline" of the volume over time.
    # We use Essentia's Envelope follower.
    envelope_algo = es.Envelope(attackTime=10, releaseTime=50) # Fast attack to catch fiddle bows
    envelope = envelope_algo(norm_audio)

    # --- 3. ARTICULATION (Staccato vs. Legato) ---
    # Logic: 
    # Legato (smooth) music has high average energy relative to its peaks.
    # Staccato (choppy) music has low average energy because there is silence between notes.
    
    env_mean = np.mean(envelope)
    env_max = np.max(envelope) if np.max(envelope) > 0 else 1.0
    
    # "Fill Ratio": How full is the sound?
    # 1.0 = Constant noise (Wall of sound) -> Very Legato
    # 0.1 = Sharp clicks -> Very Staccato
    fill_ratio = env_mean / env_max
    
    # We invert this so that higher score = More Staccato (more "articulated")
    # A value of 0.4 - 0.6 is typical for smooth folk tunes.
    # A value of 0.8+ is very choppy/plucked.
    articulation_score = np.clip(1.0 - fill_ratio, 0.0, 1.0)


    # --- 4. BOUNCINESS ---
    # Logic: Bounciness is a combination of Swing (timing) and Dynamics (pulse).
    # If a track swings BUT is totally flat dynamically (no accents), it doesn't feel "bouncy".
    # It needs both swing AND strong transient peaks.

    # Calculate "Crest Factor" (Peak-to-Average ratio) in dB
    # This measures how punchy the drums/transients are.
    rms = np.sqrt(np.mean(norm_audio**2))
    if rms > 0:
        crest_factor = 20 * np.log10(max_val / rms)
    else:
        crest_factor = 0

    # Normalize Crest Factor roughly to 0..1 range (typical music is 3dB to 20dB)
    # 3dB = flat (0.0), 20dB = very punchy (1.0)
    dynamic_punch = np.clip((crest_factor - 3) / 17, 0.0, 1.0)

    # Combine Swing Ratio (0.0-1.0 roughly) with Dynamic Punch
    # We weight swing higher because you can't be bouncy without swing.
    bounciness_score = (swing_ratio * 0.7) + (dynamic_punch * 0.3)
    bounciness_score = np.clip(bounciness_score, 0.0, 1.0)

    return {
        'articulation': float(articulation_score),
        'bounciness': float(bounciness_score),
        'dynamic_punch': float(dynamic_punch) # Useful debug metric
    }