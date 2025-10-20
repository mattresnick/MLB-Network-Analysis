import numpy as np
import networkx as nx
import pandas as pd
from springrank import SpringRank

from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as ACC


def makeGraph(filename, weights=True, val_folds=0):
    df = pd.pandas.read_csv(filename)
    edge_list = df.to_numpy()[:,1:]
    edge_list = np.array([e for e in edge_list])# if e[2]!=0])
    G = nx.DiGraph()
    
    if val_folds>0:
        full_edge_list = edge_list.copy()
        
        m = len(edge_list)
        sel_inds = np.random.choice(list(range(m)),int(m*(1-(1/val_folds))),replace=False)
        not_sel_inds = np.setdiff1d(list(range(m)), sel_inds)
        
        edge_list = edge_list[sel_inds]
        test_edges = full_edge_list[not_sel_inds]
    
    if weights: G.add_weighted_edges_from(edge_list)
    else: G.add_edges_from(edge_list[:,:2])
    
    node_list = list(G.nodes())
    
    A = nx.to_scipy_sparse_matrix(G,
                                  dtype=float,
                                  nodelist=node_list)
    
    if val_folds>0: return G, A, node_list, edge_list, test_edges
    
    return G, A, node_list, edge_list


# Get ranks using all three methods.
def getRanks(G, A, node_list):
    
    # SpringRank
    sr_rank=SpringRank(alpha=0).fit(A)
    print (type(sr_rank))
    print (sr_rank)
    sr_rank=sr_rank.ranks
    
    sr_sorted_ranks = [[node_list[i], r] for i, r in enumerate(sr_rank)]
    sr_sorted_ranks.sort(reverse=True, key=lambda x: x[1])
    
    sr_list = [sr_rank, sr_sorted_ranks]
    
    return sr_list


def getAccuracy(raw_ranks, ranks, A, test_edges):
    ranks = np.array(ranks)

    preds=[]
    obs=[]
    for i, edge in enumerate(test_edges):
        print('\r{:60}'.format('Edge #'+str(i+1)),end='')
        
        try:
            si = float(ranks[list(ranks[:,0]).index(edge[0])][1])
            sj = float(ranks[list(ranks[:,0]).index(edge[1])][1])
            
            preds.append(int(si-sj>0))
            obs.append(int(edge[2]>0))
            
        except Exception as e:
            print (e)
        
    res = [np.array(obs),np.array(preds)]
    return ACC(*res), AUC(*res)
    

print ('Making graph.')
graph_info = makeGraph('2019_batter_edges.csv', weights=True, val_folds=5)
G, A, node_list, edge_list, test_edges = graph_info

print ('Making ranks.')
spr = getRanks(G, A, node_list)
sr_raw = spr[0]
sr_sorted = spr[1]

Gacc = nx.DiGraph()
Gacc.add_edges_from(edge_list[:,:2])
node_list = list(G.nodes())

Aacc = nx.to_scipy_sparse_matrix(Gacc,
                              dtype='float64',
                              nodelist=node_list)

sr_acc, sr_auc = getAccuracy(sr_raw, sr_sorted, Aacc, test_edges)

print (f'ACC: {sr_acc}, AUC: {sr_auc}')
        



