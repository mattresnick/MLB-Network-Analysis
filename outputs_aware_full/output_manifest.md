# Output Manifest
Timestamp: 2025-10-24 01:13:01
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
- aware:batter:2024: CV orientation=as-is (p=0.978, Npos=41667)
- aware:pitcher:2024: CV orientation=as-is (p=0.983, Npos=68311)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.652, Npos=20024)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.544, Npos=23044)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.403, Npos=23610)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.470, Npos=22490)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.526, Npos=19068)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.599, Npos=19019)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.585, Npos=19862)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.488, Npos=19485)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.532, Npos=21679)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.411, Npos=19409)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.538, Npos=38604)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.665, Npos=36287)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.590, Npos=31018)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.551, Npos=35433)
- aware:pitcher:2024: OppBlock fold orientation=flipped-at-eval (p=0.454, Npos=34087)
- aware:pitcher:2024: OppBlock fold orientation=flipped-at-eval (p=0.491, Npos=33559)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.543, Npos=37541)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.575, Npos=29975)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.580, Npos=33171)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.567, Npos=31050)
- aware:batter:2024->2025: NextYear orientation=as-is (p=0.685, Npos=34260)
- aware:pitcher:2024->2025: NextYear orientation=as-is (p=0.656, Npos=45550)
