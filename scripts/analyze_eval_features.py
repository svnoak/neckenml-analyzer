#!/usr/bin/env python3
"""
Analyze feature distributions from evaluation results to guide classifier improvements.
"""
import json
from collections import defaultdict
import statistics

def analyze_features(results_file):
    with open(results_file) as f:
        data = json.load(f)

    # Group results by true style
    by_style = defaultdict(list)
    for result in data['results']:
        style = result['true_style']
        by_style[style].append(result)

    print("=" * 80)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    for style in sorted(by_style.keys()):
        tracks = by_style[style]
        print(f"\n{'='*80}")
        print(f"{style} ({len(tracks)} tracks)")
        print(f"{'='*80}")

        # Collect features
        meters = defaultdict(int)
        swings = []
        polska_scores = []
        hambo_scores = []
        ternary_confs = []

        for track in tracks:
            feat = track.get('features', {})

            # Meter
            meter = feat.get('detected_meter', 'unknown')
            meters[meter] += 1

            # Numeric features
            if feat.get('swing_ratio'):
                swings.append(feat['swing_ratio'])
            if feat.get('polska_score') is not None:
                polska_scores.append(feat['polska_score'])
            if feat.get('hambo_score') is not None:
                hambo_scores.append(feat['hambo_score'])
            if feat.get('ternary_confidence') is not None:
                ternary_confs.append(feat['ternary_confidence'])

        # Print stats
        print(f"\nMeter Distribution:")
        for meter, count in sorted(meters.items()):
            pct = (count / len(tracks)) * 100
            print(f"  {meter}: {count}/{len(tracks)} ({pct:.0f}%)")

        def print_stat(name, values):
            if values:
                print(f"\n{name}:")
                print(f"  Range: {min(values):.2f} - {max(values):.2f}")
                print(f"  Mean:  {statistics.mean(values):.2f}")
                print(f"  Median: {statistics.median(values):.2f}")

        print_stat("Swing Ratio", swings)
        print_stat("Polska Score", polska_scores)
        print_stat("Hambo Score", hambo_scores)
        print_stat("Ternary Confidence", ternary_confs)

        # Show misclassifications
        errors = [t for t in tracks if not t['is_correct']]
        if errors:
            print(f"\nMisclassifications ({len(errors)}/{len(tracks)}):")
            for err in errors:
                pred = err['predicted_style']
                conf = err.get('confidence', 0)
                feat = err.get('features', {})
                swing = feat.get('swing_ratio', 0)
                meter = feat.get('detected_meter', '?')
                polska = feat.get('polska_score', 0)
                hambo = feat.get('hambo_score', 0)
                print(f"  → {pred} (conf={conf:.2f}) | "
                      f"meter={meter} swing={swing:.2f} "
                      f"polska={polska:.2f} hambo={hambo:.2f} | "
                      f"{err['track_id']}")

    # Confusion analysis
    print("\n" + "=" * 80)
    print("CONFUSION PATTERNS")
    print("=" * 80)

    confusion_counts = defaultdict(lambda: defaultdict(int))
    for result in data['results']:
        if not result['is_correct']:
            true_style = result['true_style']
            pred_style = result['predicted_style']
            confusion_counts[true_style][pred_style] += 1

    for true_style in sorted(confusion_counts.keys()):
        print(f"\n{true_style} confused with:")
        for pred_style, count in sorted(confusion_counts[true_style].items(),
                                       key=lambda x: x[1], reverse=True):
            total = len(by_style[true_style])
            pct = (count / total) * 100
            print(f"  {pred_style}: {count}/{total} ({pct:.0f}%)")

if __name__ == "__main__":
    analyze_features("test_data/evaluation_results.json")
