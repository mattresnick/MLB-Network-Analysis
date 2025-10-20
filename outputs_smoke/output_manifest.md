# Output Manifest
Timestamp: 2025-10-15 21:17:44
Training Years: 2024
Score types: frequency
Groups: batter, pitcher
Unipartite accumulation: rate

## Methods
- Pairwise-negative AUC (original): folds=2; auto_flip=True; negatives_per_positive=1
- Other Baseline AUC: batters=(none found); pitchers=(none found)

## Output files
- validation_auc: outputs_smoke\validation_auc.csv (exists)
- validation_logloss: outputs_smoke\validation_logloss.csv
- validation_baseline_auc: outputs_smoke\validation_baseline_auc.csv (exists)
- next_year_auc: outputs_smoke\next_year_auc.csv
- iterative_compare_summary: outputs_smoke\iterative\compare_summary.csv
- rank_correlation: outputs_smoke\rank_correlation.csv
- ops_correlation: outputs_smoke\ops_correlation.csv
- levels_by_year: outputs_smoke\levels_by_year.csv
- summary_top_players: outputs_smoke\summary_top_players.csv (exists)
