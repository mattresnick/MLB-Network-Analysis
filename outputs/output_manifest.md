# Output Manifest
Timestamp: 2025-10-21 21:48:20
Training Years: 2024
Score types: frequency
Groups: batter, pitcher
Unipartite accumulation: rate

## Methods
- Pairwise-negative AUC (original): folds=5; auto_flip=True; negatives_per_positive=1
- Opponent-blockout: folds=5; heldout_fraction≈0.20
- Temperature log-loss: enabled
- Statcast log-loss: enabled
- Rank correlation: enabled; pairs=adjacent
- OPS correlation: enabled
- Other Baseline AUC: batters=OPS, WAR, xwOBA; pitchers=ERA+, FIP, K/9, WHIP, xERA, xFIP
- Statcast log-loss stats: batters=OPS, WAR, xwOBA; pitchers=ERA+, FIP, K/9, WHIP, xERA, xFIP
- Iterative ranks: passes=2; metric=sum; evaluate_auc=True
- Year-to-year validation: enabled

## Output files
- validation_auc: outputs\validation_auc.csv (exists)
- validation_logloss: outputs\validation_logloss.csv (exists)
- validation_baseline_auc: outputs\validation_baseline_auc.csv (exists)
- baseline_correlation: outputs\baseline_correlation.csv (exists)
- next_year_auc: outputs\next_year_auc.csv (exists)
- iterative_compare_summary: outputs\iterative\compare_summary.csv (exists)
- rank_correlation: outputs\rank_correlation.csv (exists)
- ops_correlation: outputs\ops_correlation.csv (exists)
- levels_by_year: outputs\levels_by_year.csv (exists)
- summary_top_players: outputs\summary_top_players.csv (exists)

## Orientation checks
- frequency:batter:2024: CV orientation=as-is (p=0.501, Npos=58677)
- frequency:pitcher:2024: CV orientation=flipped-at-eval (p=0.485, Npos=107161)
- frequency:batter:2024->2025: NextYear orientation=flipped-at-eval (p=0.500, Npos=44278)
- frequency:pitcher:2024->2025: NextYear orientation=flipped-at-eval (p=0.495, Npos=74896)
