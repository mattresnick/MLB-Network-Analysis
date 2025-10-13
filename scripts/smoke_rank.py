import os
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import springrank as s

base = os.path.dirname(os.path.dirname(__file__))
edge_path = os.path.join(base, 'At Bats', 'batter_data', 'frequency_scores', '2019_batter_edges.csv')
print('Edge path exists:', os.path.isfile(edge_path), edge_path)

df = pd.read_csv(edge_path)
edge_list = df.to_numpy()[:,1:]
G = nx.DiGraph()
G.add_weighted_edges_from(edge_list)
node_list = list(G.nodes())
try:
    A = nx.to_scipy_sparse_matrix(G, dtype=float, nodelist=node_list)
except AttributeError:
    A = nx.to_scipy_sparse_array(G, dtype=float, nodelist=node_list)
A = sp.csr_matrix(A)

model = s.SpringRank(alpha=0)
model.fit(A)
ranks = np.asarray(getattr(model, 'ranks', getattr(model, 'ranks_', None)))
if ranks is None:
    ranks = np.asarray(model.get_rescaled_ranks(target_scale=0.5))

pairs = [[node_list[i], float(ranks[i])] for i in range(len(node_list))]
pairs.sort(key=lambda x: x[1], reverse=True)
print('Top 5:', pairs[:5])
