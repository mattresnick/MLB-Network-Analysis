import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import networkx as nx
import glob

years = np.loadtxt('MLB_transactions_numbers.txt', dtype=int, usecols=0, skiprows=1)
edgelist = np.loadtxt('MLB_transactions_numbers.txt', dtype=int, usecols=(1,2), skiprows=1)
playerNames = np.genfromtxt('MLB_transactions_numbers.txt', dtype='str', usecols=3, skip_header=1)
teamNames = np.genfromtxt('MLB_team_key.txt', dtype='str', usecols=1)

batterData = []
pitcherData = []
for filename in glob.glob('ranks/*_batter_ranks_*.csv'):
    playerRanks = np.genfromtxt(filename, dtype='str', delimiter=',', usecols=(1,2), skip_header=1)
    for row in playerRanks:
        name = row[0].lower()
        name = name.replace('.','')
        name = name.replace(' ','')
        row[0] = name
    batterData.append(playerRanks)
for filename in glob.glob('ranks/*_pitcher_ranks_*.csv'):
    playerRanks = np.genfromtxt(filename, dtype='str', delimiter=',', usecols=(1,2), skip_header=1)
    for row in playerRanks:
        name = row[0].lower()
        name = name.replace('.','')
        name = name.replace(' ','')
        row[0] = name
    pitcherData.append(playerRanks)
for i in range(len(playerNames)):
    name = playerNames[i].lower()
    name = name.replace('_','')
    playerNames[i] = name
    
for year in range(10):
    bData = batterData[year]
    pData = pitcherData[year]
    bRanks = [float(row[1]) for row in bData]
    pRanks = [float(row[1]) for row in pData]
    minbRank = np.min(bRanks)
    minpRank = np.min(pRanks)
    bRanks += np.abs(minbRank)
    pRanks += np.abs(minpRank)
    for i in range(len(bData)):
        batterData[year][i][1] = bRanks[i]
    for i in range(len(pData)):
        pitcherData[year][i][1] = pRanks[i]
        
N = 30
A = np.zeros((3,10,N,N))
numMissing = np.zeros(10)
numFound = np.zeros(10)
data = [batterData, pitcherData]
for j in range(2):
    for i in range(len(edgelist)):
            t = years[i] - 2010
            if t != 10:
                name = playerNames[i]
                loc = np.where(data[j][t]==name)[0]
                if len(loc) == 0:
                    numMissing[t] += 1
                    weight = 0
                else:
                    index = loc[0]
                    weight = float(data[j][t][index][1])
                    numFound[t] += 1
                #print(loc)
                A[j][t][edgelist[i][1]][edgelist[i][0]] += weight
                A[2][t][edgelist[i][1]][edgelist[i][0]] += 1
print('Missing:', numMissing)
print('Found:  ', numFound)

year = 2015
BPtype = 1     # 0 for batters, 1 for pitchers, 2 for unweighted
t = year - 2010
k_out = np.sum(A[BPtype][t], axis=0)
nodesk = np.zeros((N,2))
for i in range(N):
    nodesk[i][0] = i
    nodesk[i][1] = k_out[i]
nodesk_sort = nodesk[nodesk[:,1].argsort()[::-1]]
nodes_sort = [row[0] for row in nodesk_sort]
labels = {}
for i in range(N):
    labels[i] = int(i)
G = nx.OrderedDiGraph()
G.add_nodes_from(range(N))
weightExp = 0.7
for i in range(N):
    for j in range(N):
        weight = A[BPtype][t][i][j]**weightExp
        if weight != 0:
            G.add_edge(j, i, weight=weight)

fig, ax = plt.subplots(figsize=(12,10))
edges = G.edges()
plt.axis('off')
pos = nx.circular_layout(G)
pos_new = dict(zip(nodes_sort, [pos[node] for node in G]))
#np.random.shuffle(pos_new)
weights = [G[u][v]['weight'] for u,v in edges]
nx.draw_networkx_nodes(G, pos_new, ax=ax, node_size=696, node_color='gold')
nx.draw_networkx_edges(G, pos_new, alpha=0.15, arrowsize=25, width=weights, connectionstyle='arc3,rad=0.314')
nx.draw_networkx_labels(G, pos_new, labels, font_size=16)
plt.show()

fig, ax = plt.subplots(figsize=(12,10))
plt.axis('off')
edges = G.edges()
pos = nx.spring_layout(G, weight='weight', iterations=500, k=10)
weights = [G[u][v]['weight'] for u,v in edges]
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=696, node_color='gold')
nx.draw_networkx_edges(G, pos, alpha=0.15, arrowsize=25, width=weights, connectionstyle='arc3,rad=0.314')
nx.draw_networkx_labels(G, pos, labels, font_size=16)
plt.show()