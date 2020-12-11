import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import scipy as sp
import glob
import SpringRank as sr

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
        name = name.replace('!','')
        row[0] = name
    batterData.append(playerRanks)
for filename in glob.glob('ranks/*_pitcher_ranks_*.csv'):
    playerRanks = np.genfromtxt(filename, dtype='str', delimiter=',', usecols=(1,2), skip_header=1)
    for row in playerRanks:
        name = row[0].lower()
        name = name.replace('.','')
        name = name.replace(' ','')
        name = name.replace('!','')
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
            weight = 0
            if len(loc) == 0:
                numMissing[t] += 1
            else:
                index = loc[0]
                weight = float(data[j][t][index][1])
                numFound[t] += 1
            #print(loc)
            A[j][t][edgelist[i][1]][edgelist[i][0]] += weight
            A[2][t][edgelist[i][1]][edgelist[i][0]] += 1
print('Missing:', numMissing)
print('Found:  ', numFound)

k_in = np.zeros((3,10,N))
k_out = np.zeros((3,10,N))
for i in range(3):
    for y in range(10):
        A1 = A[i][y]
        k_in[i][y] = np.sum(A1, axis=1)
        k_out[i][y] = np.sum(A1, axis=0)

res = 20
ks = np.linspace(0, 60, res)
cum_kin = np.zeros((3,10,res))
cum_kout = np.zeros((3,10,res))
for j in range(3):
    for y in range(10):
        for i in range(res):
            cum_kin[j][y][i] = np.count_nonzero(k_in[j][y] > ks[i])
            cum_kout[j][y][i] = np.count_nonzero(k_out[j][y] > ks[i])
            
t = 5

plt.figure(figsize=([8,6]))
plt.plot(ks, cum_kin[0][t]/N, 'o-', label='Batters')
plt.plot(ks, cum_kin[1][t]/N, 'o-', label='Pitchers')
#plt.plot(ks, cum_kin[2]/N, 'o-', label='Unweighted')
plt.xlabel(r'Incoming skill rating $k_\mathrm{in}$', fontsize=16)
plt.ylabel(r'$\mathrm{Pr}(K\geq k_\mathrm{in})$', fontsize=16)
plt.title('2015 MLB Transactions')
plt.legend()
plt.show()

k_stds = np.zeros((3,10))
for i in range(3):
    for y in range(10):
        k_stds[i][y] = np.std(k_out[i][y])
print(np.mean(k_stds[0]), np.std(k_stds[0]))
print(np.mean(k_stds[1]), np.std(k_stds[1]))

plt.figure(figsize=([8,6]))
plt.plot(ks, cum_kout[0][t]/N, 'o-', label='Batters')
plt.plot(ks, cum_kout[1][t]/N, 'o-', label='Pitchers')
plt.plot(ks, cum_kout[2][t]/N, 'o-', label='Unweighted')
plt.xlabel(r'Outgoing skill rating $k_\mathrm{out}$', fontsize=16)
plt.ylabel(r'$\mathrm{Pr}(K\geq k_\mathrm{out})$', fontsize=16)
plt.legend()
plt.show()

k_in_sort = [np.sort(k_in[i][t])[::-1] for i in range(3)]
k_out_sort = [np.sort(k_out[i][t])[::-1] for i in range(3)]
total_in = [np.sum(k_in_sort[i]) for i in range(3)]
total_out = [np.sum(k_out_sort[i]) for i in range(3)]
fracs_in = np.zeros((3,N))
fracs_out = np.zeros((3,N))
for j in range(3):
    for i in range(N):
        fracs_in[j][i] = np.sum(k_in_sort[j][:i+1]) / total_in[j]
        fracs_out[j][i] = np.sum(k_out_sort[j][:i+1]) / total_out[j]

fracs = np.linspace(0,1,N,endpoint=True)
plt.figure(figsize=([7,7]))
plt.plot(fracs, fracs_in[0], 's-', label='Batters')
plt.plot(fracs, fracs_in[1], 'o-', label='Pitchers')
#plt.plot(fracs, fracs_in[2], 'o-', label='Unweighted')
plt.plot([0,1], [0,1], '--', color='gold', label='Equality')
plt.xlabel('Fraction of teams', fontsize=16)
plt.ylabel('Fraction of incoming skill rating', fontsize=16)
plt.title('2015 MLB Transactions')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.show()

plt.figure(figsize=([7,7]))
plt.plot(fracs, fracs_out[0], 's-', label='Batters')
plt.plot(fracs, fracs_out[1], 'o-', label='Pitchers')
plt.plot(fracs, fracs_out[2], 'o-', label='Unweighted')
plt.plot([0,1], [0,1], '--', color='gold', label='Equality')
plt.xlabel('Fraction of teams', fontsize=16)
plt.ylabel('Fraction of skill rating (players)', fontsize=16)
plt.title('Outgoing')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.show()

num_in = 0
num_out = 0
denom_in = 0
denom_out = 0
Gini_in = np.zeros((3,10))
Gini_out= np.zeros((3,10))
for p in range(3):
    for y in range(10):
        for i in range(N):
            for j in range(N):
                num_in += np.abs(k_in[p][y][i] - k_in[p][y][j])
                denom_in += 2*k_in[p][y][j]
                num_out += np.abs(k_out[p][y][i] - k_out[p][y][j])
                denom_out += 2*k_out[p][y][j]
        Gini_in[p][y] = num_in / denom_in
        Gini_out[p][y] = num_out / denom_out

print('\t      Batters  \t       Pitchers')
print('G_in =   %.4f +/- %.4f, %.4f +/- %.4f' % (np.mean(Gini_in[0]), np.std(Gini_in[0]), np.mean(Gini_in[1]), np.std(Gini_in[1])))
print('G_out =  %.4f +/- %.4f, %.4f +/- %.4f' % (np.mean(Gini_out[0]), np.std(Gini_out[0]), np.mean(Gini_out[1]), np.std(Gini_out[1])))

A1 = A[1][t]
V_max = np.abs(np.transpose(sp.linalg.eigh(A1, eigvals=[N-1,N-1])[1])[0])
ranks = np.empty(shape=(N,2), dtype=object)
for i in range(N):
    ranks[i][0] = teamNames[i]
    ranks[i][1] = str(V_max[i])

listSize = 10
ranks_sort = ranks[ranks[:,1].argsort()[::-1]][:listSize]
col_width = max(len(row[0]) for row in ranks_sort) + 2  # padding
print('Team Name    Eigenvector centrality')
print('___________________________________')
for row in ranks_sort:
    print("".join(word.ljust(col_width) for word in row))
    
teamWinData = np.loadtxt('MLB_team_wins.txt', dtype=int, skiprows=2, delimiter=',')
teamWins = np.zeros((10,N))
numGames = [row[1] for row in teamWinData]
for row in teamWinData:
    for team in range(N):
        teamWins[row[0]-2010][team] = row[team+2]
        
t = 5
ranks = np.zeros((3,10,N))
for i in range(3):
    for year in range(10):
        rank = sr.get_ranks(A[i][year])
        minRank = np.min(rank)
        rank -= minRank
        ranks[i][year] = rank
        
def Sort_Tuple(tup):
    tup.sort(key = lambda x: x[1], reverse=True)  
    return tup

plt.figure(figsize=([8,6]))
plt.scatter(teamWins[t+1], ranks[1][t], marker='o', label='Pitchers')
plt.scatter(teamWins[t+1], ranks[0][t], marker='s', label='Batters')
plt.scatter(teamWins[t+1], ranks[2][t], marker='*', color='gold', label='Unweighted')
plt.title('%g MLB teams' % (t+2011))
plt.xlabel('2016 regular season wins', fontsize=16)
plt.ylabel('2015 transactions ranking', fontsize=16)
plt.legend()
plt.show()

net_degrees = np.zeros((3,10,N))
for j in range(3):
    for team in range(N):
        for y in range(10):
            for i in range(N):
                net_degrees[j][y][team] += A[j][y][team][i] - A[j][y][i][team]
                
x = teamWins[t]/numGames[t]
y0 = net_degrees[0][t]
y1 = net_degrees[1][t]
plt.figure(figsize=([8,6]))
plt.scatter(x, y0, marker='s', label='Batters')
plt.scatter(x, y1, marker='o', label='Pitchers')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y0, 1))(np.unique(x)), '--')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y1, 1))(np.unique(x)), '--')
#plt.scatter(teamWins[t], net_degrees[2][t], marker='*', color='gold', label='Unweighted')
plt.xlabel('Regular season win percentage (2015)', fontsize=16)
plt.ylabel('Net flat stat change (2015)', fontsize=16)
#plt.ylim([-50,50])
plt.legend()
plt.show()

r_xy = np.zeros((3,10))
for y in range(10):
    for i in range(3):
        num = np.sum((teamWins[y]-np.mean(teamWins[y]))*(net_degrees[i][y]-np.mean(net_degrees[i][y])))/N
        denom = np.std(teamWins[y])*np.std(net_degrees[i][y])
        r_xy[i][y] = num / denom

new_r = np.delete(r_xy[1], 1)
r = [np.mean(row) for row in r_xy]
r_err = [np.std(row) for row in r_xy]
print('Batters:    r = %.4f +/- %.4f' % (r[0], r_err[0]))
print('Pitchers:   r = %.4f +/- %.4f' % (np.mean(new_r), np.std(new_r)))
print('Unweighted: r = %.4f +/- %.4f' % (r[2], r_err[2]))

years = np.arange(2010, 2020)
years = np.delete(years, 1)
new_r = np.delete(r_xy[1], 1)

plt.figure(figsize=([8,6]))
plt.plot(range(2010,2020), r_xy[0], label='Batters')
plt.plot(years, new_r, label='Pitchers')
#plt.plot(range(2010,2020), r_xy[2], label='Unweighted')
plt.xlabel('Year (Transactions)', fontsize=16)
plt.ylabel(r'$r$', fontsize=16)
plt.legend()
plt.show()