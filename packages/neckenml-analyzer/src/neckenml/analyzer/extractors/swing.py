def calculate_swing_ratio(file_path, beat_times, return_artifacts=False):
    """
    Calculates sub-beat swing using Librosa onsets.
    Samples from multiple segments of the track for better accuracy.

    Args:
        file_path: Path to audio file
        beat_times: Beat positions in seconds
        return_artifacts: If True, returns (ratio, artifacts_dict), else just ratio
    """
    import librosa
    import numpy as np

    # Get full duration first
    duration = librosa.get_duration(path=file_path)
    
    # Define segments to sample: start (after intro), middle, and late
    # Avoid first 15s (often intro) and last 10s (often outro)
    segments = []
    if duration > 60:
        # For longer tracks, sample 3 segments
        segments = [
            (15, 45),                           # Early (skip intro)
            (duration/2 - 15, duration/2 + 15), # Middle
            (duration - 40, duration - 10)      # Late (before outro)
        ]
    elif duration > 30:
        # For medium tracks, sample from middle
        segments = [(duration/2 - 15, duration/2 + 15)]
    else:
        # Short tracks: use what we have
        segments = [(0, duration)]
    
    all_ratios = []
    all_onsets = []  # Collect all onset times for artifact storage

    for start, end in segments:
        # Clamp to valid range
        start = max(0, start)
        end = min(duration, end)
        segment_duration = end - start

        if segment_duration < 10:
            continue

        try:
            # Load segment
            y, sr = librosa.load(file_path, sr=22050, offset=start, duration=segment_duration)
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)

            # Adjust onset times to absolute position
            onsets = onsets + start
            all_onsets.extend(onsets.tolist() if hasattr(onsets, 'tolist') else list(onsets))

            # Filter beat_times to this segment
            segment_beats = [b for b in beat_times if start <= b <= end]

            if len(segment_beats) < 2:
                continue

            # Calculate ratios for this segment
            for i in range(len(segment_beats) - 1):
                beat_start = segment_beats[i]
                beat_end = segment_beats[i + 1]
                beat_duration = beat_end - beat_start

                # Find onsets in the middle of this beat interval
                candidates = [o for o in onsets if (beat_start + beat_duration * 0.2) < o < (beat_end - beat_duration * 0.2)]

                if candidates:
                    mid_point = min(candidates, key=lambda x: abs(x - (beat_start + (beat_duration / 2))))
                    first_half = mid_point - beat_start
                    second_half = beat_end - mid_point
                    if second_half > 0.001:
                        all_ratios.append(first_half / second_half)

        except Exception as e:
            print(f"[ERROR]Swing segment analysis failed: {e}")
            continue

    if not all_ratios:
        ratio = 1.0
    else:
        ratio = float(np.median(all_ratios))

    if return_artifacts:
        artifacts = {
            "librosa_onset_times": sorted(all_onsets),
            "swing_ratios": all_ratios
        }
        return ratio, artifacts

    return ratio