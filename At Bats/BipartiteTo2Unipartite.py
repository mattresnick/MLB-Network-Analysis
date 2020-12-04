import numpy as np
import networkx as nx
import os
import pandas as pd




def to2Unipartite(filename,savename1='',savename2=''):
    year=filename[:4]
    
    df = pd.pandas.read_csv(filename)
    
    # Save batter-sided edges.
    bwe = df[df.who_won == 'batter']
    bwe = bwe.loc[:,('winner','loser','score')]
    batters = np.unique(bwe.loc[:,('winner')].to_numpy())
    
    bwe = bwe.sort_values(by=['winner', 'loser'])
    bwe_arr = bwe.to_numpy()
    
    # Save pitcher-sided edges.
    pwe = df[df.who_won == 'pitcher']
    pwe = pwe.loc[:,('winner','loser','score')]
    pitchers = np.unique(pwe.loc[:,('winner')].to_numpy())
    
    pwe = pwe.sort_values(by=['winner', 'loser'])
    pwe_arr = pwe.to_numpy()
    
    # Fill in the remaining names.
    pitchers = np.unique(np.hstack((pitchers,np.unique(bwe.loc[:,('loser')].to_numpy()))))
    batters = np.unique(np.hstack((batters,np.unique(pwe.loc[:,('loser')].to_numpy()))))
    
    # Calculate group edges and save to file if filename provided.
    getGroupEdges(bwe_arr, batters, savename1)
    getGroupEdges(pwe_arr, pitchers, savename2)
    
    
    


def getGroupEdges(gwe_arr, group_players, savename=''):
    total_iter = len(group_players)**2
    current_iter = 0
    player_edgelist=[]
    for i, player1 in enumerate(group_players):
        player1_array_base = gwe_arr[gwe_arr[:,0]==player1]
        
        for j, player2 in enumerate(group_players):
            
            current_iter+=1
            if current_iter%100==0:
                progress = '%s / %s' % (str(current_iter),str(total_iter))
                print('\r{:60}'.format('Progress: '+progress),end='')
            
            if i!=j:
                # Slice out data for both players.
                player1_array = player1_array_base.copy()
                player2_array = gwe_arr[gwe_arr[:,0]==player2]
                
                # Select only scores for player 2's other group players which player 1 saw.
                player2_array = player2_array[[i for i,p in enumerate(player2_array[:,1]) if p in player1_array[:,1]]]
                
                # Select only scores for player 1's other group players which player 2 saw.
                player1_array = player1_array[[i for i,p in enumerate(player1_array[:,1]) if p in player2_array[:,1]]]
                
                # We want to subtract by adding a negative.
                player2_array[:,2] = player2_array[:,2]*(-1)
                
                # Assuming the winners and losers are sorted lexicographically, we 
                # can simply sum the scores element-wise.
                score_diffs = np.add(player1_array[:,2], player2_array[:,2])
                
                # Make negative differences zero.
                relu_diffs = np.where(score_diffs>=0, score_diffs, 0)
                
                # Sum up scores for all other group players matchups to be the edge weight.
                total_score = np.sum(relu_diffs)
                
                player_edgelist.append([player1,player2,total_score])
    print()
    player_edges_df = pd.DataFrame(player_edgelist, columns = ['winner', 'loser', 'score'])
    
    player_edges_df.to_csv(savename)


def forceNoParallel(savename, filename='', df=None):
    if len(filename)>0:
        df = pd.pandas.read_csv(filename)
    edge_list = df.to_numpy()[:,1:]
    edge_list = np.array([e for e in edge_list if e[2]!=0])
    
    reduced_edgelist=[]
    for i,edge in enumerate(edge_list):
        print('\r{:60}'.format('Edge #'+str(i+1)+'/'+str(len(edge_list))),end='')
        
        player2_edges = edge_list[edge_list[:,0]==edge[1]]
        player2_edges = player2_edges[player2_edges[:,1]==edge[0]]
        
        if len(player2_edges)>0:
            
            #print (str(edge)+' - '+str(player2_edges[0]))
            
            new_score = edge[2]-player2_edges[0][2]
            
            if edge[2]>player2_edges[0][2]:
                reduced_edgelist.append([edge[0],edge[1],new_score])
            elif edge[2]<player2_edges[0][2]:
                reduced_edgelist.append([edge[0],edge[1],0])
            else:
                reduced_edgelist.append(edge)
                reduced_edgelist.append(player2_edges[0])
                
    player_edges_df = pd.DataFrame(reduced_edgelist, columns = ['winner', 'loser', 'score'])
    player_edges_df.to_csv(savename)



for year in range(2009,2020):
    print ('\n'+str(year))
    
    filename='./general_data/handmade/'+str(year)+'_edges_only.csv'
    
    savename1=str(year)+'_batter_edges.csv'
    savename2=str(year)+'_pitcher_edges.csv'
    
    to2Unipartite(filename,savename1,savename2)
    
    new_filename = './batter_data/handmade_scores/'+savename1
    forceNoParallel(savename=new_filename,filename=savename1)
    
    filename=str(year)+'_pitcher_edges.csv'
    new_filename = './pitcher_data/handmade_scores/'+savename2
    forceNoParallel(savename=new_filename,filename=savename2)



for year in range(2009,2020):
    print ('\n'+str(year))
    
    filename='./general_data/frequency/'+str(year)+'_edges_only.csv'
    
    save_dir = './intermediate_results/frequency/'
    savename1=str(year)+'_batter_edges.csv'
    savename2=str(year)+'_pitcher_edges.csv'
    
    to2Unipartite(filename,save_dir+savename1,save_dir+savename2)
    
    new_filename = './batter_data/frequency_scores/'+savename1
    forceNoParallel(savename=new_filename,filename=save_dir+savename1)
    
    filename=str(year)+'_pitcher_edges.csv'
    new_filename = './pitcher_data/frequency_scores/'+savename2
    forceNoParallel(savename=new_filename,filename=save_dir+savename2)


pt_dir = './general_data/pitch_type/'
for year in range(2009,2020):
    
    for pt in os.listdir(pt_dir):
        if pt[-4:]!='.csv':
            print ('\n'+str(year) + ' ('+pt+')')
            
            filename=pt_dir+pt+'/'+str(year)+'_edges_only.csv'
            
            save_dir = './intermediate_results/pitch_type/'+pt+'/'
            savename1=str(year)+'_batter_edges.csv'
            savename2=str(year)+'_pitcher_edges.csv'
            
            to2Unipartite(filename,save_dir+savename1,save_dir+savename2)
            
            new_filename = './batter_data/pitchtype_scores/'+pt+'/'+savename1
            forceNoParallel(savename=new_filename,filename=save_dir+savename1)
            
            filename=str(year)+'_pitcher_edges.csv'
            new_filename = './pitcher_data/pitchtype_scores/'+pt+'/'+savename2
            forceNoParallel(savename=new_filename,filename=save_dir+savename2)


inn_dir = './general_data/inning/'
for year in range(2009,2020):
    print ('\n'+str(year))
    for inn in os.listdir(inn_dir):
        if inn[-4:]!='.csv' and inn!='10':
            
            print ('\n'+str(year) + ' ('+inn+')')
            
            filename=inn_dir+inn+'/'+str(year)+'_edges_only.csv'
            
            save_dir = './intermediate_results/inning/'+inn+'/'
            savename1=str(year)+'_batter_edges.csv'
            savename2=str(year)+'_pitcher_edges.csv'
            
            to2Unipartite(filename,save_dir+savename1,save_dir+savename2)
            
            new_filename = './batter_data/inning_scores/'+inn+'/'+savename1
            forceNoParallel(savename=new_filename,filename=save_dir+savename1)
            
            filename=str(year)+'_pitcher_edges.csv'
            new_filename = './pitcher_data/inning_scores/'+inn+'/'+savename2
            forceNoParallel(savename=new_filename,filename=save_dir+savename2)




