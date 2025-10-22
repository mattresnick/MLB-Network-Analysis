# Configs guide (consolidated)

Primary config you should use for full runs:

- example_2024.json — aware-only, shrink+harmonic+tether, CSV-only outputs, and all four validations enabled:
  - edge_block, pa_block, loeo (under `validation.modes`)
  - opponent-block (under `validation.extra.opponent_blockout`)

How to vary runs:
- Years: edit `pipeline.years` (e.g., [2023, 2024]); no separate year configs required.
- Scrape: control via `pipeline.scrape` (if present) or leave to default logic; no separate noscrape configs required.
- K-sweep: toggle `analysis.k_sweep.enabled` and `only` inside this same config when you need it.

Deprecated/legacy configs (kept temporarily):
- example_aware_*.json, example_*_sum.json, example_*_force*.json, example_*_regen.json, example_*_pa.json, example_*_minimal.json, example_*_quick.json, example_smoke_baseline.json, example_frequency_cv.json

These are no longer necessary. Use `example_2024.json` and flip flags/years as needed. If you want, we can delete them in a follow-up cleanup once you confirm nothing in your workflow still references them.
