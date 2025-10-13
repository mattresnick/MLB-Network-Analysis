# MLB-Network-Analysis
Analysis of various types of networks given 11 years of MLB play-level data.

The code an data can be found in the relevant folders for each section of the report. We also include
the presentation slides, the report write-up itself, and an appendix. The appendix is not necessary to
the write-up, and only consists of extra information and plots that was not essential to the report but
may have been of interest.

If you have questions about reading/running the code, please reach out via email.

At-bat code and analysis: Matt

Trade code and analysis: Yifan

## JSON Config Driven Pipeline (New)

You can now generate new player ranking outputs (SpringRank-based orderings and optional scaled ELO-like levels) via a single JSON configuration file without manually editing the original research scripts. PageRank and BiRank have been deprecated in this pipeline due to inferior performance in this domain.

### 1. Create / Edit a Config

See `config/example_minimal.json` for a minimal run and `config/example_full.json` for all options. A draft schema is documented in `config/schema.md`.

Minimal example:
```
{
	"pipeline": {
		"years": [2019],
		"allow_2020": false,
		"dry_run": false,
		"score_types": ["handmade"],
		"paths": {"output_dir": "outputs"}
	}
}
```

### 2. Install Dependencies

Ensure you have the required Python packages (see `requirements.txt`). Create a virtual environment if desired.

### 3. Run the Pipeline

From the project root (PowerShell on Windows):
```
python run_config.py --config config/example_minimal.json
```

### 4. Outputs & Artifacts

Results are written under the directory specified by `pipeline.paths.output_dir` (default `outputs/`). The pipeline auto-generates missing edge files for requested `score_types` (handmade, frequency, pitch_type, inning) directly from raw at-bat data; then builds unipartite player-vs-player edge lists and computes SpringRank.

Multi-format output: specify `pipeline.output.formats` (any of `csv`, `parquet`, `json`). Files are produced with matching extensions. Core artifacts include:

Per year / condition / group:
- `<score_type>/<group>[/<pitch_type>|/<inning>]/<year>_springrank.*` raw SpringRank
- `<year>_springrank_scaled.*` scaled rank (if enabled)

Aggregates:
- `summary_top_players.*` top N summary across all processed spans
- `levels_by_year.*` scalar range for scaled ranks (if enabled)
- `validation_report.*` nodes/edges/density per graph
- `validation_auc.*` cross-validated Accuracy and AUC per year/condition (if validation_folds > 0)
- `mobility_report.*` quartile transition mobility (optional)
- `anomalies_report.*` anomalous rank deltas (optional)
- `rolling_summary.*` rolling window span metadata (optional)

Caching: A manifest (`outputs/manifest.json`) stores file signatures and config signature; unchanged inputs are skipped on subsequent runs.

### 5. Extending / Notes

Key capabilities now implemented:

- Auto edge generation (handmade, frequency, pitch_type, inning)
- Handedness filters (`filters.stand`, `filters.p_throws`)
- SpringRank only (deprecated PageRank/BiRank paths removed from new pipeline)
- Scaled ranks and levels aggregation (`levels_by_year`)
- Multi-format outputs (CSV / Parquet / JSON)
- Caching with file + config signatures
- Mobility analysis (quartile transitions) `analysis.mobility.enabled`
- Anomaly detection (YOY large scaled rank deltas) `analysis.anomalies.*`
- Rolling window rankings `analysis.rolling.enabled` with `windows`
- CLI subcommands (`scrape`, `edges`, `rank`, `full`)
- FastAPI microservice (rank & player endpoints)
- Docker image & CI workflow

Season 2020 is excluded by default unless `allow_2020: true`.

### 6. CLI Usage

```
python cli.py --config config/example_full.json scrape   # just scrape
python cli.py --config config/example_full.json edges    # build edges/unipartite
python cli.py --config config/example_full.json rank     # compute rankings only
python cli.py --config config/example_full.json full     # full pipeline
```

### 7. FastAPI Service

Run locally (after generating outputs):
```
uvicorn api:app --reload
```
Endpoints: `/ranks`, `/top`, `/player/{name}`, `/mobility`, `/anomalies`, `/health`.

### 8. Docker

Build & run API (serving existing `outputs/`):
```
docker build -t mlb-net .
docker run -p 8000:8000 -v %CD%/outputs:/app/outputs mlb-net
```
(On Linux/Mac replace `%CD%` with `$(pwd)`.)

### 9. CI

GitHub Actions workflow runs lint (syntax compile) and pytest (if tests exist) on pushes / PRs to main lines.

### 10. Tests

Lightweight tests (expand as needed):
```
pytest -q
```

### 11. Roadmap / Future Enhancements

- Streamlit UI integration for config upload & execution
- Parallel ranking computation
- Expanded anomaly strategies (z-score, rolling volatility)
- Additional player metadata enrichment
- Automated weekly data update job

Contributions welcome – open an issue or PR.

## Running in VS Code (shortcuts)

- First-time setup: use the launcher "First run: setup env + full (MLB)". This provisions a repo-local .venv and installs dependencies.
- Day-to-day: use "Full pipeline: example_frequency_cv (MLB)" or the targeted launchers:
	- "Scrape: 2024 only (MLB)" — fetch just 2024 raw data
	- "Full pipeline: 2024 only (MLB)" — run edges + ranks just for 2024

