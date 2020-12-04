import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Mobility between the lower and upper half of levels.
def avgMobilityHalf(s_ranks1,s_ranks2):
    levels1 = np.max(s_ranks1[:,1]) - np.min(s_ranks1[:,1])
    levels2 = np.max(s_ranks2[:,1]) - np.min(s_ranks2[:,1])
    
    mid1= np.min(s_ranks1[:,1])+(levels1/2)
    mid2= np.min(s_ranks2[:,1])+(levels2/2)
    
    upward_mobility, downward_mobility = [], []
    total_overlap=0
    for i,(player, rank1) in enumerate(s_ranks1):
        
        if player in s_ranks2[:,0]:
            total_overlap+=1
            
            ind = list(s_ranks2[:,0]).index(player)
            rank2 = s_ranks2[ind,1]
            
            
            if rank1<mid1 and rank2>mid2:
                upward_mobility.append(1)
            elif rank1<mid1 and rank2<mid2:
                upward_mobility.append(0)
            elif rank1>mid1 and rank2<mid2:
                downward_mobility.append(1)
            elif rank1>mid1 and rank2>mid2:
                downward_mobility.append(0)
    
    return np.mean(upward_mobility), np.mean(downward_mobility)


# Mobility between the upper and lower quartiles of levels.
def avgMobilityQuart(s_ranks1,s_ranks2):
    levels1 = np.max(s_ranks1[:,1]) - np.min(s_ranks1[:,1])
    levels2 = np.max(s_ranks2[:,1]) - np.min(s_ranks2[:,1])
    
    quart1_1 = np.min(s_ranks1[:,1])+(levels1/4)
    quart1_2 = np.max(s_ranks1[:,1])-(levels1/4)
    
    quart2_1 = np.min(s_ranks2[:,1])+(levels2/4)
    quart2_2 = np.max(s_ranks2[:,1])-(levels2/4)
    
    upward_mobility, downward_mobility = [], []
    for i,(player, rank1) in enumerate(s_ranks1):
        
        if player in s_ranks2[:,0]:
            
            
            ind = list(s_ranks2[:,0]).index(player)
            rank2 = s_ranks2[ind,1]
            
            
            if rank1<quart1_1:
                if rank2>quart2_2:
                    upward_mobility.append(1)
                else:
                    upward_mobility.append(0)
            elif rank1>quart1_2:
                if rank2<quart2_1:
                    downward_mobility.append(1)
                else:
                    downward_mobility.append(0)
    
    return np.mean(upward_mobility), np.mean(downward_mobility)




def processMobility(timeframe):
    
    mobility_stats=[]
    for year in range(2009,2020-timeframe):
        year_stats=[]
        for group in ['batter','pitcher']:
            
            #print('Year: '+str(year)+'. Group: '+group)
            
            filedir1 = './'+group+'_data/handmade_scores/scaled_ranks_'+str(year)+'.csv'
            ranks1 = pd.read_csv(filedir1).to_numpy()[:,1:]
            
            filedir2 = './'+group+'_data/handmade_scores/scaled_ranks_'+str(year+timeframe)+'.csv'
            ranks2 = pd.read_csv(filedir2).to_numpy()[:,1:]
            
            up_m, down_m = avgMobilityQuart(ranks1, ranks2)
            year_stats.extend([np.round(up_m,4), np.round(down_m,4)])
        mobility_stats.append(year_stats)
        
    return np.array(mobility_stats)


tframes = [1,2,3,4,5,6,7,8,9]
all_stats = []
for tframe in tframes:
    stats = np.nan_to_num(processMobility(tframe))
    all_stats.append(np.mean(stats,axis=0))
all_stats = np.array(all_stats)

colors=plt.cm.rainbow(np.linspace(0,1,4))

fig, ax = plt.subplots()
ax.set(ylabel='Average Percent of Players',xlabel='Look-Ahead Timeframe (years)')
ax.plot(tframes,all_stats[:,0],color=colors[0],label='Batter Upward Mobility')
ax.plot(tframes,all_stats[:,1],color='grey',label='Batter Downward Mobility')
ax.plot(tframes,all_stats[:,2],color=colors[2],label='Pitcher Upward Mobility')
ax.plot(tframes,all_stats[:,3],color=colors[3],label='Pitcher Downward Mobility')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('skillmobility',dpi=200)




