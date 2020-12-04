import numpy as np
import Rankings as r
import matplotlib.pyplot as plt

'''
Produces plots for cumulative degree distributions for batter and pitcher
networks over the years for both in and out degrees.
'''

def degreeDistribution(filename,direction):
    graph_info = r.makeGraph(filename, weights=True)
    G, A, node_list, edge_list = graph_info
    
    #degree_list = np.array([G.in_degree[n] for n in node_list])
    
    if direction=='in': dim = 0
    else: dim=1
    
    degree_list = np.sum(A,axis=dim)
    
    return degree_list.ravel()





years = range(2009,2020)
all_dists = []
for year in years:
    for group in ['batter','pitcher']:
        
        printstring = 'Year: '+str(year)+'. Group: '+group
        print('\r{:100}'.format(printstring),end='')
        
        filename='./'+group+'_data/handmade_scores/'+str(year)+'_'+group+'_edges.csv'
        writedir = './'+group+'_data/handmade_scores/'
        
        all_dists.append(degreeDistribution(filename,'out'))

fig, ax = plt.subplots()
colors=plt.cm.rainbow(np.linspace(0,1,len(years)))
for i, dist in enumerate(all_dists):
    if i%2==0:
        
        dist=np.array([d for d in dist.flatten()])
        
        values, counts = np.unique(dist, return_counts=True)
        all_vals = list(zip(values, counts))
        all_vals.sort(key=lambda x: x[0])
        all_vals = np.array(all_vals)
        
        total=np.sum(all_vals[:,1])
        
        x=all_vals[:,0]
        y=[np.sum(all_vals[j:,1])/total for j in range(len(all_vals))]
        
        label=str(years[int(i/2)])
        color=colors[int(i/2)]
        ax.plot(x,y, label=label, color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(xlabel='Batter Out-degree',
               ylabel='Proportion of Batters')


ax.legend()
plt.tight_layout()
plt.show()
#fig.savefig('batter_out_ccdf',dpi=200)

fig, ax = plt.subplots()
colors=plt.cm.rainbow(np.linspace(0,1,len(years)))
for i, dist in enumerate(all_dists):
    if i%2!=0:
        
        dist=np.array([d for d in dist.flatten()])
        
        values, counts = np.unique(dist, return_counts=True)
        all_vals = list(zip(values, counts))
        all_vals.sort(key=lambda x: x[0])
        all_vals = np.array(all_vals)
        
        total=np.sum(all_vals[:,1])
        
        x=all_vals[:,0]
        y=[np.sum(all_vals[j:,1])/total for j in range(len(all_vals))]
        
        label=str(years[int(i/2)])
        color=colors[int(i/2)]
        ax.plot(x,y, label=label, color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(xlabel='Pitcher Out-degree',
               ylabel='Proportion of Pitchers')


ax.legend()
plt.tight_layout()
plt.show()
#fig.savefig('pitcher_out_ccdf',dpi=200)









years = range(2009,2020)
all_dists = []
for year in years:
    for group in ['batter','pitcher']:
        
        printstring = 'Year: '+str(year)+'. Group: '+group
        print('\r{:100}'.format(printstring),end='')
        
        filename='./'+group+'_data/handmade_scores/'+str(year)+'_'+group+'_edges.csv'
        writedir = './'+group+'_data/handmade_scores/'
        
        all_dists.append(degreeDistribution(filename,'in'))

fig, ax = plt.subplots()
colors=plt.cm.rainbow(np.linspace(0,1,len(years)))
for i, dist in enumerate(all_dists):
    if i%2==0:
        
        dist=np.array([d for d in dist.flatten()])
        
        values, counts = np.unique(dist, return_counts=True)
        all_vals = list(zip(values, counts))
        all_vals.sort(key=lambda x: x[0])
        all_vals = np.array(all_vals)
        
        total=np.sum(all_vals[:,1])
        
        x=all_vals[:,0]
        y=[np.sum(all_vals[j:,1])/total for j in range(len(all_vals))]
        
        label=str(years[int(i/2)])
        color=colors[int(i/2)]
        ax.plot(x,y, label=label, color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(xlabel='Batter In-degree',
               ylabel='Proportion of Batters')


ax.legend()
plt.tight_layout()
plt.show()
#fig.savefig('batter_in_ccdf',dpi=200)




fig, ax = plt.subplots()
colors=plt.cm.rainbow(np.linspace(0,1,len(years)))
for i, dist in enumerate(all_dists):
    if i%2!=0:
        
        dist=np.array([d for d in dist.flatten()])
        
        values, counts = np.unique(dist, return_counts=True)
        all_vals = list(zip(values, counts))
        all_vals.sort(key=lambda x: x[0])
        all_vals = np.array(all_vals)
        
        total=np.sum(all_vals[:,1])
        
        x=all_vals[:,0]
        y=[np.sum(all_vals[j:,1])/total for j in range(len(all_vals))]
        
        label=str(years[int(i/2)])
        color=colors[int(i/2)]
        ax.plot(x,y, label=label, color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set(xlabel='Pitcher In-degree',
               ylabel='Proportion of Pitchers')


ax.legend()
plt.tight_layout()
plt.show()
#fig.savefig('pitcher_in_ccdf',dpi=200)


