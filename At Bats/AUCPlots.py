import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


colors=plt.cm.rainbow(np.linspace(0,1,3))

# Plot of handmade scores.
for group in ['batter','pitcher']:
    filedir = './'+group+'_data/handmade_scores/ranks_by_year.csv'
    data = pd.read_csv(filedir).to_numpy()[:,1:]
    x = data[:,0]
    
    fig, ax = plt.subplots()
    ax.plot(x,data[:,4],color=colors[0],label='SpringRank AUC')
    ax.plot(x,data[:,5],color=colors[1],label='PageRank AUC')
    ax.plot(x,data[:,6],color=colors[2],label='BiRank AUC')
    plt.legend()
    plt.show()
    
    print (str(np.mean(data[:,4]))+' - '+ str(np.std(data[:,4])))
    print (str(np.mean(data[:,5]))+' - '+ str(np.std(data[:,5])))
    print (str(np.mean(data[:,6]))+' - '+ str(np.std(data[:,6])))


# Plot of frequency scores.
for group in ['batter','pitcher']:
    filedir = './'+group+'_data/frequency_scores/ranks_by_year.csv'
    data = pd.read_csv(filedir).to_numpy()[:,1:]
    x = data[:,0]
    
    fig, ax = plt.subplots()
    ax.plot(x,data[:,4],color=colors[0],label='SpringRank AUC')
    ax.plot(x,data[:,5],color=colors[1],label='PageRank AUC')
    ax.plot(x,data[:,6],color=colors[2],label='BiRank AUC')
    plt.legend()
    plt.show()
    
    print (str(np.mean(data[:,4]))+' - '+ str(np.std(data[:,4])))
    print (str(np.mean(data[:,5]))+' - '+ str(np.std(data[:,5])))
    print (str(np.mean(data[:,6]))+' - '+ str(np.std(data[:,6])))



# Plot of pitch type scores.
pitch_types=['CH','CU','FC','FF','FS','FT','SI','SL']
colors=plt.cm.rainbow(np.linspace(0,1,len(pitch_types)))
for group in ['batter','pitcher']:
    fig, ax = plt.subplots()
    for i, pt in enumerate(pitch_types):
        filedir = './'+group+'_data/pitchtype_scores/'+pt+'/ranks_by_year.csv'
        data = pd.read_csv(filedir).to_numpy()[:,1:]
        ax.plot(data[:,0],data[:,1],color=colors[i],label=pitch_types[i])
        
        print (pt+': '+str(np.mean(data[:,1]))+' - '+ str(np.std(data[:,1])))
    
    plt.legend()
    plt.show()


# Plot of inning scores.
innings=list(range(1,10))
colors=plt.cm.rainbow(np.linspace(0,1,len(innings)))
for group in ['batter','pitcher']:
    fig, ax = plt.subplots()
    for i, inn in enumerate(innings):
        filedir = './'+group+'_data/inning_scores/'+str(inn)+'/ranks_by_year.csv'
        data = pd.read_csv(filedir).to_numpy()[:,1:]
        ax.plot(data[:,0],data[:,1],color=colors[i],label=str(innings[i]))
        
        print (str(inn)+': '+str(np.mean(data[:,1]))+' - '+ str(np.std(data[:,1])))
    
    plt.legend()
    plt.show()