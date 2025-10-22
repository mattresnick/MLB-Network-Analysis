# Output Manifest
Timestamp: 2025-10-21 23:58:21
Training Years: 2024
Score types: aware
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
- Year-to-year validation: enabled

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
- aware:batter:2024: CV orientation=as-is (p=0.977, Npos=41667)
- aware:pitcher:2024: CV orientation=as-is (p=0.983, Npos=68311)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.441, Npos=41414)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.615, Npos=45509)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.419, Npos=39038)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.580, Npos=40474)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.517, Npos=40611)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.559, Npos=69859)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.588, Npos=66680)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.529, Npos=65576)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.589, Npos=72588)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.603, Npos=66022)
- aware:batter:2024->2025: NextYear orientation=as-is (p=0.687, Npos=34260)
- aware:pitcher:2024->2025: NextYear orientation=as-is (p=0.662, Npos=45550)
