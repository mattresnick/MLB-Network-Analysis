import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import brentq
import os

from SpringRank import SpringRank as sr
import Rankings as r

def get_scaled_ranks(A, ranks, a, b, scale=0.75):
    """
    Modified from original SpringRank package (I didn't want ranks to be 
    calculated every time since I'm assuming I already have them).
    """
    inverse_temperature = brentq(sr.eqs39, a, b, args=(ranks, A))
    scaling_factor = 1 / (np.log(scale / (1 - scale)) / (2 * inverse_temperature))
    scaled_ranks = sr.scale_ranks(ranks,scaling_factor)
    return scaled_ranks



def complexityWriter(filename,writedir,year,hd=False):
    graph_info = r.makeGraph(filename, weights=True)
    G, A, node_list, edge_list = graph_info
    
    sr_raw,sr_sorted = r.getSpringRank(A, node_list)
    
    sr_scaled = get_scaled_ranks(A, sr_raw, 0.01, 20)
    sr_levels = max(sr_scaled) - min(sr_scaled)
    
    results = np.array([year, sr_levels])
    col_list = ['Year','SpringRank Levels']
    
    writefile = 'levels_by_year.csv'
    
    if writefile in os.listdir(writedir):
        data = pd.pandas.read_csv(writedir+writefile).to_numpy()[:,1:]
        all_data = np.vstack((data, results))
        write_df = pd.DataFrame(all_data, columns=col_list)
        write_df.to_csv(writedir+writefile)
    else:
        write_df = pd.DataFrame([results], columns=col_list)
        write_df.to_csv(writedir+writefile)
    
    if hd:
        save_ranks = np.array([[node_list[i], r] for i, r in enumerate(sr_scaled)])
        write_df = pd.DataFrame(save_ranks, columns=['Player','Rank'])
        write_df.to_csv(writedir+'scaled_ranks_'+str(year)+'.csv')




#Get levels by year for each of batters and pitchers.
for year in range(2019,2008,-1):
    for group in ['batter','pitcher']:
        
        printstring = 'Year: '+str(year)+'. Group: '+group
        print('\r{:100}'.format(printstring),end='')
        
        filename='./'+group+'_data/handmade_scores/'+str(year)+'_'+group+'_edges.csv'
        writedir = './'+group+'_data/handmade_scores/'
        
        complexityWriter(filename,writedir,year,hd=True)




#Get levels by year for each of batters and pitchers for frequency scores.
for year in range(2019,2008,-1):
    for group in ['batter','pitcher']:
        
        printstring = 'Year: '+str(year)+'. Group: '+group
        print('\r{:100}'.format(printstring),end='')
        
        filename='./'+group+'_data/frequency_scores/'+str(year)+'_'+group+'_edges.csv'
        writedir = './'+group+'_data/frequency_scores/'
        
        complexityWriter(filename,writedir,year)




#Get levels by year for each of batters and pitchers for pitch type.
pitch_types=['CH','CU','FC','FF','FS','FT','SI','SL']

for pt in pitch_types:
    for year in range(2019,2008,-1):
        for group in ['batter','pitcher']:
            
            printstring = 'Year: '+str(year)+'. Group: '+group+'. Pitch type: '+pt
            print('\r{:100}'.format(printstring),end='')
            
            filename='./'+group+'_data/pitchtype_scores/'+pt+'/'+str(year)+'_'+group+'_edges.csv'
            writedir = './'+group+'_data/pitchtype_scores/'+pt+'/'
            
            complexityWriter(filename,writedir,year)






#Get levels by year for each of batters and pitchers for inning.
innings=list(range(1,10))
for inn in innings:
    for year in range(2019,2008,-1):
        for group in ['batter','pitcher']:
            
            printstring = 'Year: '+str(year)+'. Group: '+group+'. Inning: '+str(inn)
            print('\r{:100}'.format(printstring),end='')
            
            filename='./'+group+'_data/inning_scores/'+str(inn)+'/'+str(year)+'_'+group+'_edges.csv'
            writedir = './'+group+'_data/inning_scores/'+str(inn)+'/'
            
            complexityWriter(filename,writedir,year)


