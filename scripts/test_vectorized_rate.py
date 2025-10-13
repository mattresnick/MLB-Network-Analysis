import os
import sys
sys.path.insert(0, os.getcwd())
import pandas as pd
from pipeline import _unipartite_vectorized

year = 2024
raw_dir = os.path.join('At Bats','general_data')
edge_only = os.path.join(raw_dir, 'frequency', f'{year}_edges_only.csv')
print('[test] reading', edge_only)
df = pd.read_csv(edge_only)
# Batter group
bwe = df[df.who_won=='batter'][['winner','loser','score']].sort_values(['winner','loser'])
pwe = df[df.who_won=='pitcher'][['winner','loser','score']].sort_values(['winner','loser'])
print('[test] rows: batter', len(bwe), 'pitcher', len(pwe))
print('[test] computing batter vectorized rate...')
bv = _unipartite_vectorized(bwe, metric='rate', year=year, raw_data_dir=raw_dir, winners_role='batter')
print('[test] batter edges (vectorized-rate):', len(bv))
print(bv.head())
print('[test] computing pitcher vectorized rate...')
pv = _unipartite_vectorized(pwe, metric='rate', year=year, raw_data_dir=raw_dir, winners_role='pitcher')
print('[test] pitcher edges (vectorized-rate):', len(pv))
print(pv.head())
