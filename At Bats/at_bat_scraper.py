import baseball_scraper as pyb


date_ranges = [
['2019-03-20','2019-10-30'],
['2018-03-29','2018-10-28'],
['2017-04-02','2017-11-01'],
['2016-04-03','2016-11-02'],
['2015-04-05','2015-11-01'],
['2014-03-22','2014-10-29'],
['2013-03-31','2013-10-30'],
['2012-03-28','2012-10-28'],
['2011-03-31','2011-10-28'],
['2010-04-04','2010-11-01'],
['2009-04-05','2009-11-04']]


for drange in date_ranges:
    print (drange[0][:4])
    # Retreive game data.
    data = pyb.statcast(*drange)
    #data = pyb.statcast_single_game(529429)
    
    # Only look at AB ending events.
    data = data.dropna(subset=['events']) 
    
    # Categories to pull.
    categories = ['pitch_type','player_name','batter','events','description',
                  'home_team','away_team','inning','stand','p_throws','home_score',
                  'away_score']
    
    
    
    # Convert IDs to integers.
    data.astype({'batter': 'int32'}).dtypes
    
    # Pull relevant data and delete origianl data.
    all_data = data[categories]
    #del data
    
    # Get a list of pitcher IDs so we can have their names.
    player_ids = [int(n) for n in all_data[categories[2]].to_numpy()]
    retrieved_names = pyb.playerid_reverse_lookup(player_ids, key_type='mlbam')
    batter_names = retrieved_names.loc[:,('key_mlbam','name_first','name_last')]
    #del retrieved_names
    
    # Captialize the pitcher names and combine them.
    batter_names.loc[:,('name_first')] = batter_names.loc[:,('name_first')].str.capitalize() 
    batter_names.loc[:,('name_last')]  = batter_names.loc[:,('name_last')].str.capitalize() 
    batter_names.loc[:,('batter_name')] = batter_names.loc[:,('name_first', 'name_last')].agg(' '.join, axis=1)
    
    # Merge both tables into one so the pitchers' names are there and the IDs are not.
    merged = all_data.join(batter_names[['key_mlbam', 'batter_name']].set_index('key_mlbam'), on='batter')
    del merged['batter']

    
    merged.to_csv('at_bat_data_'+drange[0][:4]+'.csv')
