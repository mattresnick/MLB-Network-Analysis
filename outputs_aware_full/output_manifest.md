# Output Manifest
Timestamp: 2025-10-20 02:30:43
Training Years: 2024, 2025
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
- aware:batter:2024: CV orientation=as-is (p=0.977, Npos=41667)
- aware:batter:2025: CV orientation=as-is (p=0.998, Npos=8329)
- aware:pitcher:2024: CV orientation=as-is (p=0.983, Npos=68311)
- aware:pitcher:2025: CV orientation=as-is (p=0.988, Npos=180499)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.605, Npos=45838)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.516, Npos=37688)
- aware:batter:2024: OppBlock fold orientation=as-is (p=0.527, Npos=42879)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.468, Npos=41005)
- aware:batter:2024: OppBlock fold orientation=flipped-at-eval (p=0.463, Npos=40280)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.638, Npos=70679)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.557, Npos=70003)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.557, Npos=64614)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.545, Npos=67343)
- aware:pitcher:2024: OppBlock fold orientation=as-is (p=0.570, Npos=67261)
- aware:batter:2025: OppBlock fold orientation=flipped-at-eval (p=0.469, Npos=212615)
- aware:batter:2025: OppBlock fold orientation=flipped-at-eval (p=0.453, Npos=206528)
- aware:batter:2025: OppBlock fold orientation=flipped-at-eval (p=0.390, Npos=198291)
- aware:batter:2025: OppBlock fold orientation=flipped-at-eval (p=0.410, Npos=197842)
- aware:batter:2025: OppBlock fold orientation=flipped-at-eval (p=0.457, Npos=202146)
- aware:pitcher:2025: OppBlock fold orientation=as-is (p=0.612, Npos=183699)
- aware:pitcher:2025: OppBlock fold orientation=as-is (p=0.622, Npos=177882)
- aware:pitcher:2025: OppBlock fold orientation=as-is (p=0.635, Npos=183257)
- aware:pitcher:2025: OppBlock fold orientation=as-is (p=0.652, Npos=173537)
- aware:pitcher:2025: OppBlock fold orientation=as-is (p=0.641, Npos=182775)
