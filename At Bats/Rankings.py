import numpy as np
import networkx as nx
import os
import pandas as pd
from SpringRank import SpringRank as sr
import birankpy as br
from copy import copy
from scipy.optimize import brentq
from math import cosh, copysign


def getGroups(filename, group='batter'):
    # Returns the batter/pitcher specific data.
    # Edges where only that group won and the names.
    
    df = pd.pandas.read_csv(filename)
    
    group_wins_edges = df[df.who_won == group]
    group_wins_edges = group_wins_edges.loc[:,('winner','loser','score')]
    
    return group_wins_edges



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
        


# Only get ranks via SpringRank
def getSpringRank(A, node_list):
    
    
    sr_rank=sr.SpringRank(A, alpha=0)
    sr_sorted_ranks = [[node_list[i], r] for i, r in enumerate(sr_rank)]
    sr_sorted_ranks.sort(reverse=True, key=lambda x: x[1])
    
    return sr_rank, sr_sorted_ranks


# Get ranks using all three methods.
def getRanks(G, A, node_list, filename, birank_group, weights=True):
    
    # SpringRank
    sr_rank=sr.SpringRank(A, alpha=0)
    sr_sorted_ranks = [[node_list[i], r] for i, r in enumerate(sr_rank)]
    sr_sorted_ranks.sort(reverse=True, key=lambda x: x[1])
    
    sr_list = [sr_rank, sr_sorted_ranks]
    
    # PageRank
    pr_rank = nx.pagerank(G)
    pr_sorted_ranks = [[item[0],item[1]] for item in pr_rank.items()]
    pr_sorted_ranks.sort(reverse=True, key=lambda x: x[1])
    
    pr_list = [np.array(list(pr_rank.values())), pr_sorted_ranks]
    
    # BiRank
    group_edges = getGroups(filename, birank_group)
    bn = br.BipartiteNetwork()
    
    if weights:
        bn.set_edgelist(group_edges, 
                        top_col='winner', 
                        bottom_col='loser',
                        weight_col='score')
    else:
        bn.set_edgelist(group_edges, 
                        top_col='winner', 
                        bottom_col='loser')
    
    top_birank_df, bottom_birank_df = bn.generate_birank()
    br_sorted_ranks = list([list(pair) for pair in top_birank_df.to_numpy()])
    br_rank = copy(br_sorted_ranks)
    br_sorted_ranks.sort(reverse=True, key=lambda x: x[1])
    
    br_list = [np.array(np.array(br_rank)[:,1], np.float32), br_sorted_ranks]
    
    return sr_list, pr_list, br_list




def eqs36(beta, s, A):
    N = A.shape[0]
    x = 0
    for i in range(N):
        for j in range(N):
            if A[i, j] == 0:
                continue
            else:
                sign_term = copysign(1,A[i, j]-(A[i, j]+A[j, i])*(1/(1+np.exp(-2*beta*(s[i] - s[j])))))
                x += ((A[i, j]+A[j, i])*(s[i] - s[j])/(cosh(2*beta*(s[i] - s[j]))+1))*sign_term
    return x















