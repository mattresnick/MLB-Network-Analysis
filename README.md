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

Ensure you have the required Python packages (original environment requirements + `baseball_scraper`, `networkx`, `SpringRank`, `birankpy`, `pandas`, `numpy`, `scipy`). Create a virtual environment if desired.

### 3. Run the Pipeline

From the project root (PowerShell on Windows):
```
python run_config.py --config config/example_minimal.json
```

### 4. Outputs

Results are written under the directory specified by `pipeline.paths.output_dir` (default `outputs/`). The pipeline auto-generates missing edge files for requested `score_types` (handmade, frequency, pitch_type, inning) directly from raw at-bat data, then builds unipartite player-vs-player edge lists and computes SpringRank.

For each year and group (batter/pitcher) you will see (example for handmade):

- `handmade/<group>/<year>_springrank.csv` : Raw SpringRank ordering.
- `handmade/<group>/<year>_springrank_scaled.csv` : Scaled ranks (if `scale_ranks` true).
- `summary_top_players.csv` : Aggregated top N (configurable via `ranking.top_n`).

### 5. Extending / Notes

Auto edge generation is now supported for all listed score types. The scraper automatically downloads raw at-bat data for requested seasons if missing, unless `scrape.force` is true (then it re-downloads). Season 2020 is excluded by default unless `allow_2020: true` is set under `pipeline`.

### 6. Tests

Lightweight config validation tests can be run with:
```
python tests_config.py
```

### 7. Roadmap

- Additional filter application (stand / p_throws) to be extended to edge generation (placeholder parsed now).
- Caching & incremental updates for future new seasons.
- Provide Parquet / JSON output formats.
- Performance tuning for large multi-condition runs.

Feel free to open an issue or PR with improvements.

