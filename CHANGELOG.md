# Changelog

Record of classifier and feature-extraction changes: what changed, why,
and what was tried and rejected. Code comments should stay short and
point here for the full story, not carry the history inline.

## Unreleased

### Changed

- `punchiness` now uses coefficient of variation (`std/mean`) of
  per-beat energy instead of raw mean energy. The old formula
  (`tanh((sum/len)*10)`) saturated to exactly 1.0 for almost every real
  track, because per-beat energy is an unbounded raw signal that scales
  with recording loudness, not a normalized 0-1 value. Verified against
  100 stored tracks: old formula returned exactly 1.0 for 100/100; new
  formula returns a real spread (min 0.32, max 0.99, mean 0.67).
  Fixed in `neckenml-core/reanalysis.py`,
  `neckenml-analyzer/extractors/rhythm.py`, and mirrored in the local
  trainer tool.

- Triplet grouping for `r1_mean`/`r2_mean`/`r3_mean` and
  `downbeat_dominance` is now phase-aligned: grouping starts from the
  first beat actually marked as the downbeat (`beat_positions == 1`),
  not always from index 0. The old index-0 assumption was wrong on
  roughly 44% of a sampled track set, rotating the Polska/Hambo
  signature for those tracks. `beat_positions` was already being stored;
  this only changes how it's used.
  Fixed in `neckenml-core/reanalysis.py` and
  `neckenml-analyzer/extractors/rhythm.py`.

- Classification embedding is now 213-dimensional (was 217).
  `rms`, `zcr`, `onset_rate`, and `punchiness` describe a track's feel
  (jumpy vs. smooth), not its dance style, and diluted the
  style-discrimination signal by being mixed into the same vector. They
  are now reported separately as `feel_profile` in the analysis result
  and excluded from the vector fed to `ClassificationHead`.
  `ClassificationHead.FEATURE_VERSION` bumped 4 -> 5,
  `EXPECTED_FEATURE_COUNT` 217 -> 213. Any model trained on the old shape
  is auto-detected as stale on load and cleared, per the existing
  version-check in `ClassificationHead._load()`.
  Requires a reanalysis pass to regenerate stored embeddings, and a
  retrain, before this takes effect on live predictions.

- `_extract_lightweight_features()` (the source of `rms`/`zcr`/
  `onset_rate`) now logs the real exception on failure instead of
  silently returning `[0.0, 0.0, 0.0]`. Confirmed against 100 stored
  tracks that all three values are exactly 0.0 for 100/100 -- meaning
  the exception handler is being hit on every call, not as an edge
  case. Root cause not yet found: Essentia isn't available in the
  environment this was diagnosed in, so the real exception couldn't be
  reproduced directly. Check worker logs after the next real run.

### Fixed

- `ClassificationHead.train()` filtered `embeddings`/`labels` into
  `valid_embeddings`/`valid_labels` by vector length, then trained on
  the unfiltered originals anyway. Any batch containing vectors of
  mixed lengths would silently build a ragged array and corrupt the
  fit. Now trains on the filtered lists.

- `StyleClassifier._get_secondary_styles()` never suggested Menuett as
  a secondary style under any condition, so a Vals track whose swing
  already sat in Menuett's own accepted range had no path to ever
  surface as a Menuett candidate. Vals and Menuett are structurally
  close (both near-even beats, first-beat accent) and external sources
  describe them as adjacent forms, so added a Vals<->Menuett secondary
  suggestion, gated on the existing swing-band thresholds. Deliberately
  did not add Polska<->Menuett: that boundary is a documented historical
  tune-book relabeling pattern (see `research/obp.0314.pdf` in the
  dansbart repo), not a living-practice ambiguity like Vals<->Menuett.

### Tried and rejected

- `RandomForestClassifier(class_weight='balanced')`, to address severe
  label imbalance (347 Polska vs. 4 Menuett in one training run, ~87:1).
  Measured worse on a held-out split, not better: accuracy dropped from
  58% to 34%, weighted-F1 from 0.63 to 0.47, while macro-F1 stayed flat
  (0.34 -> 0.34). At this imbalance ratio, `'balanced'` dilutes
  confidence broadly enough that even majority-class predictions
  (Polska, Vals) drop below `predict()`'s fixed 0.4 confidence
  threshold into "Unknown," without making minority classes confident
  enough to make up for it. Do not reapply without also reconsidering
  that fixed threshold.
