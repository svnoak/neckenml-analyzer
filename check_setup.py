#!/usr/bin/env python3
"""
Setup validation script for neckenml-analyzer testing framework.

Checks that all necessary components are in place and provides
helpful feedback for missing items.

Usage:
    python check_setup.py
"""

import sys
from pathlib import Path

# Try to import yaml, but don't fail if it's missing (we'll check it later)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def check_file(path: Path, description: str, required: bool = True) -> bool:
    """Check if a file exists."""
    exists = path.exists()
    status = "✓" if exists else ("✗" if required else "⚠")
    req_text = "(required)" if required else "(optional)"

    print(f"{status} {description}: {path} {req_text}")

    if not exists and required:
        return False
    return True


def check_directory(path: Path, description: str, required: bool = True) -> bool:
    """Check if a directory exists and optionally report contents."""
    exists = path.exists() and path.is_dir()
    status = "✓" if exists else ("✗" if required else "⚠")
    req_text = "(required)" if required else "(optional)"

    if exists:
        count = len(list(path.glob('*')))
        print(f"{status} {description}: {path} {req_text} - {count} files")
    else:
        print(f"{status} {description}: {path} {req_text} - NOT FOUND")

    if not exists and required:
        return False
    return True


def check_yaml_content(path: Path) -> tuple:
    """Check YAML file is valid and return content."""
    if not YAML_AVAILABLE:
        print(f"  ⚠ Cannot validate YAML - PyYAML not installed")
        return False, {}

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return True, data
    except Exception as e:
        print(f"  ⚠ Error reading YAML: {e}")
        return False, {}


def check_dependencies():
    """Check Python dependencies."""
    print("\n" + "="*70)
    print("CHECKING PYTHON DEPENDENCIES")
    print("="*70)

    all_ok = True

    # Required packages
    required_packages = [
        ('yaml', 'PyYAML'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pandas', 'pandas'),
    ]

    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"✗ {package_name} NOT installed")
            print(f"  Install with: pip install {package_name}")
            all_ok = False

    # neckenml (critical)
    try:
        from neckenml.analyzer import AudioAnalyzer
        from neckenml.classifier import StyleClassifier
        print(f"✓ neckenml installed and importable")
    except ImportError as e:
        print(f"✗ neckenml NOT installed or not importable")
        print(f"  Error: {e}")
        print(f"  Install with: pip install -e /path/to/neckenml")
        all_ok = False

    return all_ok


def check_file_structure():
    """Check test data file structure."""
    print("\n" + "="*70)
    print("CHECKING FILE STRUCTURE")
    print("="*70)

    root = Path.cwd()
    all_ok = True

    # Core files
    all_ok &= check_file(root / "evaluate_classification.py", "Evaluation script")
    all_ok &= check_file(root / "visualize_confusion_matrix.py", "Visualization script")
    all_ok &= check_file(root / "threshold_tuning.ipynb", "Tuning notebook")
    all_ok &= check_file(root / "check_setup.py", "Setup checker")

    # Test data directory
    test_data = root / "test_data"
    all_ok &= check_directory(test_data, "Test data directory")

    if test_data.exists():
        all_ok &= check_file(test_data / "test_tracks.yaml", "Test tracks definition")
        all_ok &= check_file(test_data / "known_issues.yaml", "Known issues tracker")
        all_ok &= check_file(test_data / "README.md", "Test data README")

        # Generated files (optional)
        check_file(test_data / "evaluation_results.json", "Evaluation results", required=False)
        check_file(test_data / "baseline_results.json", "Baseline results", required=False)

        # Audio directories
        audio_dir = test_data / "audio"
        check_directory(audio_dir, "Audio directory", required=False)

        if audio_dir.exists():
            styles = ['polska', 'polka', 'hambo', 'vals', 'schottis', 'snoa',
                     'engelska', 'slangpolska', 'ganglat']

            for style in styles:
                style_dir = audio_dir / style
                check_directory(style_dir, f"  {style.capitalize()} audio", required=False)

    return all_ok


def check_test_tracks():
    """Check test_tracks.yaml content."""
    print("\n" + "="*70)
    print("CHECKING TEST TRACKS CONFIGURATION")
    print("="*70)

    test_tracks_path = Path("test_data/test_tracks.yaml")

    if not test_tracks_path.exists():
        print("✗ test_tracks.yaml not found - skipping validation")
        return False

    valid, data = check_yaml_content(test_tracks_path)

    if not valid:
        return False

    tracks = data.get('tracks', [])
    print(f"\n✓ YAML is valid - found {len(tracks)} track definitions")

    if len(tracks) == 0:
        print("⚠ No tracks defined yet - you need to add test tracks!")
        print("  See test_data/README.md for instructions")
        return False

    # Analyze tracks
    style_counts = {}
    missing_files = []
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}

    for track in tracks:
        # Count by style
        style = track.get('true_style', 'Unknown')
        style_counts[style] = style_counts.get(style, 0) + 1

        # Check file exists
        file_path = Path(track.get('file_path', ''))
        if not file_path.exists():
            missing_files.append(str(file_path))

        # Count difficulty
        difficulty = track.get('difficulty', 'medium')
        if difficulty in difficulty_counts:
            difficulty_counts[difficulty] += 1

    # Report findings
    print(f"\nTracks by style:")
    for style, count in sorted(style_counts.items()):
        status = "✓" if count >= 4 else "⚠"
        target = "(target: 4-5)" if count < 4 else ""
        print(f"  {status} {style}: {count} tracks {target}")

    print(f"\nTracks by difficulty:")
    for difficulty, count in difficulty_counts.items():
        print(f"  {difficulty}: {count} tracks")

    if missing_files:
        print(f"\n⚠ Warning: {len(missing_files)} audio files not found:")
        for i, path in enumerate(missing_files[:5]):
            print(f"  - {path}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        return False
    else:
        print(f"\n✓ All {len(tracks)} audio files found!")

    # Critical styles check
    critical_styles = ['Polska', 'Polka', 'Hambo', 'Vals']
    missing_critical = [s for s in critical_styles if s not in style_counts]

    if missing_critical:
        print(f"\n⚠ Missing critical styles: {', '.join(missing_critical)}")
        print("  These are the most important for classification accuracy")
        return False

    return True


def print_next_steps():
    """Print helpful next steps."""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)

    print("""
1. Add test audio files:
   - Copy audio files to test_data/audio/<style>/ folders
   - Aim for 4-5 tracks per style minimum
   - Start with: Polska, Polka, Hambo, Vals

2. Update test_tracks.yaml:
   - Add entry for each test track
   - Specify true_style, difficulty, and characteristics
   - See test_data/README.md for template

3. Run baseline evaluation:
   python evaluate_classification.py --verbose

4. Generate visualizations:
   python visualize_confusion_matrix.py

5. Analyze in notebook:
   jupyter notebook threshold_tuning.ipynb

6. Document issues:
   - Update test_data/known_issues.yaml
   - Track threshold changes and their impact

See test_data/README.md for detailed workflow guide.
""")


def main():
    print("="*70)
    print("NECKENML-ANALYZER TESTING FRAMEWORK SETUP CHECK")
    print("="*70)

    # Check all components
    deps_ok = check_dependencies()
    structure_ok = check_file_structure()
    tracks_ok = check_test_tracks()

    # Overall status
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if deps_ok and structure_ok and tracks_ok:
        print("\n✅ All checks passed! You're ready to start testing.")
        print("\nRun: python evaluate_classification.py --verbose")
    else:
        print("\n⚠️  Some issues found. Review the output above.")

        if not deps_ok:
            print("\n❌ Python dependencies missing - install required packages")

        if not structure_ok:
            print("\n❌ File structure incomplete - check missing files")

        if not tracks_ok:
            print("\n❌ Test tracks not configured - add audio files and update YAML")

        print_next_steps()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
