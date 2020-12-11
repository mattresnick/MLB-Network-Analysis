# Team-level player transactions
This folder contains the analysis code and player ranks/stats to analyze player contract trades among MLB teams from 2010-2019.

## remove_accents.py
Replaces all accented characters with their non-accented equivalents in associated stats and ranks data.

## visual.py
Produces network visualizations of team-level transactions, weighted by player skill-ratings

## analyze_ranks.py
Analyzes degree distributions and network structure of team-level transactions weighted by player skill-rating from SpringRank with the handmade scoring scheme. Compares net skill rating change with regular season win percentage.

## analyze_flat.py
Analyzes degree distributions and network structure of team-level transactions weighted by batter OPS and pitcher WHIP. Compares net flat stat change with regular season win percentage.
