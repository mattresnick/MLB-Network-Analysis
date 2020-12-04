'''
Fix up the pitch type and inning edges so they are easier to process later on.
'''

import numpy as np
import pandas as pd
import os
'''
# Write to pitch type folders
pitch_types=['CH','CU','EP','FC','FF','FO','FS','FT','KC','KN','SI','SL']
directory ='./general_data/pitch_type/'
for file in os.listdir(directory):
    if file[-4:]=='.csv':
        df = pd.pandas.read_csv(directory+file)
        data = df.to_numpy()
        
        for pt in pitch_types:
            write_data = data[data[:,0]==pt]
            write_data = write_data[:,1:]
            write_df = pd.DataFrame(write_data, columns = ['winner', 'loser','who_won', 'score'])
            write_df.to_csv(directory+pt+'/'+file)
'''

# Write to inning folders, where 10+ innings just go to 10.
innings=list(range(1,11))
directory ='./general_data/inning/'
for file in os.listdir(directory):
    if file[-4:]=='.csv':
        df = pd.pandas.read_csv(directory+file)
        data = df.to_numpy()
        
        for inn in innings:
            if inn!=10:
                write_data = data[data[:,0]==inn]
            else:
                write_data = data[data[:,0]>=inn]
            write_data = write_data[:,1:]
            write_df = pd.DataFrame(write_data, columns = ['winner', 'loser','who_won', 'score'])
            write_df.to_csv(directory+str(inn)+'/'+file)
        
        