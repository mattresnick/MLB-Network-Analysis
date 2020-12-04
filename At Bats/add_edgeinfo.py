import numpy as np
import networkx as nx
import pandas as pd
import os


class Scorer:
    def __init__(self, b_scoring, p_scoring):
        self.b_dict = b_scoring
        self.p_dict = p_scoring
        
    def scoreEvent(self, in_str):
        batter,pitcher,event = in_str.split(',')
        
        # Batter-sided scoring.
        if event in list(self.b_dict.keys()):
            s = self.b_dict[event]
            
        # Pitcher-sided scoring.
        elif event in list(self.p_dict.keys()):
            s = (-1)*self.p_dict[event]
            
        # Some other event.
        else:
            return 0
        
        if s<0: return [pitcher, batter, np.abs(s),'pitcher']
        else: return [batter, pitcher,s,'batter']


# Add edge data to AB data.
def addEdgeInfotoRaw(filename, scorer):
    print (filename)
    #filename='at_bat_data_2019.csv'
    df = pd.pandas.read_csv(filename)
    
    df.loc[:,('edge_comb')] = df.loc[:,('batter_name', 'player_name','events')].agg(','.join, axis=1)
    df['edge_info'] = [scorer.scoreEvent(quad) for quad in df['edge_comb']]
    
    df = df[df.edge_info != 0]
    
    df.loc[:,('winner')] = [val[0] for val in df['edge_info']]
    df.loc[:,('loser')] = [val[1] for val in df['edge_info']]
    df.loc[:,('score')] = [val[2] for val in df['edge_info']]
    df.loc[:,('who_won')] = [val[3] for val in df['edge_info']]
    
    del df['edge_comb']
    del df['edge_info']
    df.to_csv(filename, index=False)

# Create files with just edge info.
def onlyEdges(filename, savename, pt=False, inn=False):
    print (filename)
    df = pd.pandas.read_csv(filename)
    
    if pt:
        edge_only_df = df.loc[:,('pitch_type','winner','loser','score', 'who_won')]
        edge_only_df = edge_only_df.groupby(['pitch_type','winner', 'loser', 'who_won']).sum()
    elif inn:
        edge_only_df = df.loc[:,('inning','winner','loser','score', 'who_won')]
        edge_only_df = edge_only_df.groupby(['inning','winner', 'loser', 'who_won']).sum()
    else:
        edge_only_df = df.loc[:,('winner','loser','score', 'who_won')]
        edge_only_df = edge_only_df.groupby(['winner', 'loser', 'who_won']).sum()
    
    edge_only_df.to_csv(savename)


# Get frequency of AB result types.
def resultFrequency(filename, group_wins_scoring, group='batter'):
    df = pd.pandas.read_csv(filename)
    all_results = df['events'].to_numpy()
    
    group_wins_results = list(group_wins_scoring.keys())
    
    # Count total appearances of each result.
    group_scoring = {}
    for result in group_wins_results:
        total = len(all_results[all_results==result])
        group_scoring[result] = total
    
    # Count total appearances of all results together.
    total_results = 0
    for val in list(group_scoring.values()):
        total_results+=val
    
    # Divide aggregate total by result total. Rarer events = bigger score.
    for item in group_scoring.items():
        scale = group_wins_scoring[item[0]]
        group_scoring[item[0]] = scale*(item[1]/total_results)
    
    return group_scoring
    
    



'''
# Scores for batter-sided AB wins.
base_batter_scoring = {'hit_by_pitch':1,
                       'walk':2,
                       'single':2,
                       'double':7,
                       'triple':75,
                       'home_run':20}

# Scores for pitcher-sided AB wins.
base_pitcher_scoring = {'fielders_choice':1,
                       'fielders_choice_out':1,
                       'other_out':1,
                       'field_out':1,
                       'force_out':2,
                       'grounded_into_double_play':2,
                       'strikeout':6}



# Scores by frequency of occurance.
for year in range(2009,2020):
    print (year)
    
    filename='./general_data/at_bat_data_'+str(year)+'.csv'
    savename = './general_data/frequency/'+str(year)+'_edges_only.csv'
    
    batter_scoring = resultFrequency(filename, 
                                     base_batter_scoring, 
                                     group='batter')
    
    pitcher_scoring = resultFrequency(filename, 
                                     base_pitcher_scoring, 
                                     group='pitcher')
    
    
    scorer = Scorer(batter_scoring, pitcher_scoring)
    addEdgeInfotoRaw(filename, scorer)
    onlyEdges(filename, savename)

'''





# Scores for batter-sided AB wins.
base_batter_scoring = {'hit_by_pitch':1,
                       'walk':2,
                       'single':3,
                       'double':6,
                       'triple':9,
                       'home_run':12}

# Scores for pitcher-sided AB wins.
base_pitcher_scoring = {'fielders_choice':1,
                       'fielders_choice_out':1,
                       'other_out':1,
                       'field_out':1,
                       'force_out':2,
                       'grounded_into_double_play':2,
                       'strikeout':6}



'''
# Handcrafted scores.
for year in range(2009,2020):
    print (year)
    
    filename='./general_data/at_bat_data_'+str(year)+'.csv'
    savename = './general_data/handmade/'+str(year)+'_edges_only.csv'
    
    scorer = Scorer(base_batter_scoring, base_pitcher_scoring)
    addEdgeInfotoRaw(filename, scorer)
    onlyEdges(filename,savename)





# Handcrafted scores with pitch type included.
for year in range(2009,2020):
    print (year)
    
    filename='./general_data/at_bat_data_'+str(year)+'.csv'
    savename = './general_data/pitch_type/'+str(year)+'_edges_only.csv'
    
    scorer = Scorer(base_batter_scoring, base_pitcher_scoring)
    addEdgeInfotoRaw(filename, scorer)
    onlyEdges(filename,savename,pt=True)


'''


# Handcrafted scores with inning included.
for year in range(2009,2020):
    print (year)
    
    filename='./general_data/at_bat_data_'+str(year)+'.csv'
    savename = './general_data/inning/'+str(year)+'_edges_only.csv'
    
    scorer = Scorer(base_batter_scoring, base_pitcher_scoring)
    addEdgeInfotoRaw(filename, scorer)
    onlyEdges(filename,savename,inn=True)




