import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

colors=plt.cm.rainbow(np.linspace(0,1,3))

# Plot of handmade scores.
for group in ['batter','pitcher']:
    filedir = './'+group+'_data/handmade_scores/levels_by_year.csv'
    data = pd.read_csv(filedir).to_numpy()[:,1:]
    x = data[:,0]
    
    fig, ax = plt.subplots()
    ax.plot(x,data[:,1],color=colors[0],label='SpringRank Skill Levels')
    plt.legend()
    plt.show()
    
    print ('Handmade: '+str(np.round(np.mean(data[:,1]),2))+' - '+ \
           str(np.round(np.std(data[:,1]),2)))


# Plot of frequency scores.
for group in ['batter','pitcher']:
    filedir = './'+group+'_data/frequency_scores/levels_by_year.csv'
    data = pd.read_csv(filedir).to_numpy()[:,1:]
    x = data[:,0]
    
    fig, ax = plt.subplots()
    ax.plot(x,data[:,1],color=colors[0],label='SpringRank Skill Levels')
    plt.legend()
    plt.show()
    
    print ('Frequency: '+str(np.round(np.mean(data[:,1]),2))+' - '+ \
           str(np.round(np.std(data[:,1]),2)))



# Plot of pitch type scores.
pitch_types=['CH','CU','FC','FF','FS','FT','SI','SL']
colors=plt.cm.rainbow(np.linspace(0,1,len(pitch_types)))
for group in ['batter','pitcher']:
    fig, ax = plt.subplots()
    for i, pt in enumerate(pitch_types):
        filedir = './'+group+'_data/pitchtype_scores/'+pt+'/levels_by_year.csv'
        data = pd.read_csv(filedir).to_numpy()[:,1:]
        ax.plot(data[:,0],data[:,1],color=colors[i],label=pitch_types[i])
        
        print (pt+': '+str(np.round(np.mean(data[:,1]),2))+' - '+ \
               str(np.round(np.std(data[:,1]),2)))
    
    plt.legend()
    plt.show()


# Plot of inning scores.
innings=list(range(1,10))
colors=plt.cm.rainbow(np.linspace(0,1,len(innings)))
for group in ['batter','pitcher']:
    fig, ax = plt.subplots()
    for i, inn in enumerate(innings):
        filedir = './'+group+'_data/inning_scores/'+str(inn)+'/levels_by_year.csv'
        data = pd.read_csv(filedir).to_numpy()[:,1:]
        ax.plot(data[:,0],data[:,1],color=colors[i],label=str(innings[i]))
        
        print (str(inn)+': '+str(np.round(np.mean(data[:,1]),2))+' - '+ \
               str(np.round(np.std(data[:,1]),2)))
    
    plt.legend()
    plt.show()







for group in ['batter','pitcher']:
    filedir = './'+group+'_data/handmade_scores/scaled_ranks_2019.csv'
    data = pd.read_csv(filedir).to_numpy()[:,1:]
    data = np.array(data[:,1],dtype = 'float')
    
    fig, ax = plt.subplots()
    
    ax2 = ax.twinx()
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(alpha=0.75,linestyle=(0,(5,10)))
    
    ax2.set_ylabel('Count',color='firebrick')
    ax2.hist(data,bins=40,color='firebrick',rwidth=0.75,zorder=1)
    
    ax.set(xlabel='Scaled Rank',ylabel='Density')
    density = gaussian_kde(data)
    x = np.linspace(np.min(data),np.max(data),400)
    density.covariance_factor = lambda: 0.25
    density._compute_covariance()
    ax.plot(x,density(x),color='black',zorder=10)
    
    if group=='batter':
        plt.title('2019 Batting Complexity')
    else:
        plt.title('2019 Pitching Complexity')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(group+'_complexitydensity_2019',dpi=200)
    







