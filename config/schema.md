JSON Config Schema (Draft)

Top-level keys:

pipeline:
  years: [int, ...] inclusive years list OR object {start: int, end: int}
  score_types: ["handmade"|"frequency"|"pitch_type"|"inning"] (multi allowed)
  pitch_types: ["CH","CU","FC","FF","FS","FT","SI","SL"] (if score_types includes pitch_type)
  innings: [1-9] (if score_types includes inning)
  filters:
    stand: ["L","R"] optional
    p_throws: ["L","R"] optional
  algorithms: {springrank: true, pagerank: false, birank: false}
  validation_folds: int (0 for none)
  ranking:
    top_n: int (default 25)
    scale_ranks: bool (produce scaled ranks / levels)
    output_levels: bool (write levels_by_year.csv)
  scrape:
    force: false (re-scrape even if csv exists)
    chunk_size: optional (future)
  paths:
    raw_data_dir: "At Bats/general_data"
    batter_dir: "At Bats/batter_data"
    pitcher_dir: "At Bats/pitcher_data"
    intermediate_dir: "At Bats/intermediate_results"
    output_dir: "outputs" (created if missing)
  output:
    formats: ["csv"] future: json, parquet
    include_edge_lists: true
    include_rank_tables: true
    include_scaled_rank_tables: true (if scale_ranks)
logging:
  level: INFO|DEBUG
  progress: true

Example minimal config in example_minimal.json.
