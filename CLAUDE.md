# neckenml-analyzer

Audio analysis and dance style classification library for Swedish folk music. The repository contains three packages.

| Package | Path | License | Purpose |
|---|---|---|---|
| `neckenml-core` | `packages/neckenml-core/` | MIT | Classification from stored features |
| `neckenml-analyzer` | `packages/neckenml-analyzer/` | AGPL-3.0 | Audio analysis with Essentia and madmom |
| `neckenml` | `packages/neckenml/` | AGPL-3.0 | Meta-package |

Consumers: the dansbart.se feature worker uses `neckenml-core`. `dansbart-audio-worker` uses `neckenml-analyzer`.

## Status

The library is dormant. The dansbart.se workers still use it. Start new classifier work only when the maintainer asks for it. Before you diagnose a classifier bug, read `planning/14-classification-fixes-and-learnings.md` in the dansbart workspace.

## Rules

- Keep `neckenml-core` MIT-compatible. Do not import `neckenml.analyzer`, Essentia, madmom, or librosa in `neckenml-core`.
- The feature vector is built in three places. Change all three together:
  - `packages/neckenml-analyzer/src/neckenml/analyzer/audio_analyzer.py`
  - `packages/neckenml-core/src/neckenml/core/reanalysis.py`
  - `ml-trainer/tuner/engine.py` in the dansbart workspace
- To find every place, search for `EXPECTED_FEATURE_COUNT`, `full_vector`, and `folk_vector_list`.
- `ml-trainer` installs `neckenml-core` in editable mode from this checkout. An edit here changes `ml-trainer` immediately.

## Commands

```bash
pip install -e "packages/neckenml-core[dev]"
pytest tests/core/
pytest tests/analyzer/ -m "not requires_audio"
black --check packages/ tests/
flake8 packages/ tests/
```

## Releases

Push a tag to publish to PyPI:

- `core-v1.2.3`: `neckenml-core` only.
- `analyzer-v1.2.3`: `neckenml-analyzer` only.
- `v1.2.3`: `neckenml` meta-package only.
- `release-v1.2.3`: all packages.

## Conventions

- Commit format: `<type>: <summary>`. Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `style`.
- No emojis in code, comments, or commit messages.
