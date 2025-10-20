# Output Manifest
Timestamp: 2025-10-18 05:36:55
Training Years: 2024
Score types: aware
Groups: batter, pitcher
Unipartite accumulation: sum

## Methods
- Pairwise-negative AUC (original): folds=5; auto_flip=True; negatives_per_positive=1
- Opponent-blockout: folds=5; heldout_fraction≈0.20
- Temperature log-loss: enabled
- Statcast log-loss: enabled
- Other Baseline AUC: batters=OPS, WAR, xwOBA; pitchers=ERA+, FIP, K/9, WHIP, xERA, xFIP
- Statcast log-loss stats: batters=OPS, WAR, xwOBA; pitchers=ERA+, FIP, K/9, WHIP, xERA, xFIP

## Output files
- validation_auc: outputs_aware_full\validation_auc.csv (exists)
- validation_logloss: outputs_aware_full\validation_logloss.csv (exists)
- validation_baseline_auc: outputs_aware_full\validation_baseline_auc.csv (exists)
- baseline_correlation: outputs_aware_full\baseline_correlation.csv (exists)
- next_year_auc: outputs_aware_full\next_year_auc.csv (exists)
- iterative_compare_summary: outputs_aware_full\iterative\compare_summary.csv
- rank_correlation: outputs_aware_full\rank_correlation.csv
- ops_correlation: outputs_aware_full\ops_correlation.csv (exists)
- levels_by_year: outputs_aware_full\levels_by_year.csv (exists)
- summary_top_players: outputs_aware_full\summary_top_players.csv (exists)

## Orientation checks
- aware:batter:2024: CV orientation=as-is (p=0.521, Npos=41667)
- aware:pitcher:2024: CV orientation=as-is (p=0.616, Npos=68311)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.504, Npos=42546)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.475, Npos=43513)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.575, Npos=39938)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.542, Npos=41028)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.495, Npos=40665)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.553, Npos=67283)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.577, Npos=71190)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.565, Npos=66727)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.604, Npos=68024)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.541, Npos=67501)
