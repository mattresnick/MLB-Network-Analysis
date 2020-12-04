import numpy as np
import networkx as nx
import os
import pandas as pd
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as ACC

from SpringRank import SpringRank as sr
import Rankings as r


# Switch to AUC?
def getAccuracy(raw_ranks, ranks, A, test_edges,a=0.01,b=20, auc=True):
    ranks = np.array(ranks)
    if auc:
        preds=[]
        obs=[]
        for i, edge in enumerate(test_edges):
            print('\r{:60}'.format('Edge #'+str(i+1)),end='')
            
            try:
                si = float(ranks[list(ranks[:,0]).index(edge[0])][1])
                sj = float(ranks[list(ranks[:,0]).index(edge[1])][1])
                
                preds.append(int(si-sj>0))
                obs.append(int(edge[2]>0))
                
            except Exception as e:
                print (e)
            
        res = [np.array(obs),np.array(preds)]
        return ACC(*res), AUC(*res)
    
    else:
        correct = []
        for i, edge in enumerate(test_edges):
            print('\r{:60}'.format('Edge #'+str(i+1)),end='')
            
            try:
                si = float(ranks[list(ranks[:,0]).index(edge[0])][1])
                sj = float(ranks[list(ranks[:,0]).index(edge[1])][1])
                
                pred = int(si-sj>0)
                if (edge[2]==0 and pred==0) or (edge[2]>0 and pred>0):
                    correct.append(1)
                else:
                    correct.append(0)
                
            except Exception as e:
                print (e)
        
        accuracy = np.mean(correct)
        return accuracy


'''
Get ranks + accuracies by year for each of batters and pitchers for each rank type.
'''
for year in range(2009,2020):
    for group in ['batter','pitcher']:
        
        filename1='./'+group+'_data/handmade_scores/'+str(year)+'_'+group+'_edges.csv'
        
        print ('Making graph.')
        graph_info = r.makeGraph(filename1, weights=True, val_folds=5)
        G, A, node_list, edge_list, test_edges = graph_info
        
        filename2='./general_data/handmade/'+str(year)+'_edges_only.csv'
        
        print ('Making ranks.')
        spr, pr, bir = r.getRanks(G, A, node_list, filename2, birank_group=group, weights=True)
        sr_raw, pr_raw, br_raw = spr[0], pr[0], bir[0]
        sr_sorted, pr_sorted, br_sorted = spr[1], pr[1], bir[1]
        
        Gacc = nx.DiGraph()
        Gacc.add_edges_from(edge_list[:,:2])
        node_list = list(G.nodes())
        
        Aacc = nx.to_scipy_sparse_matrix(Gacc,
                                      dtype='float64',
                                      nodelist=node_list)
        
        sr_acc, sr_auc = getAccuracy(sr_raw, sr_sorted, Aacc, test_edges)
        pr_acc, pr_auc = getAccuracy(pr_raw, pr_sorted, Aacc, test_edges)
        br_acc, br_auc = getAccuracy(br_raw, br_sorted, Aacc, test_edges)
        results = np.array([year, sr_acc, pr_acc, br_acc, sr_auc, pr_auc, br_auc])
        col_list = ['Year','SpringRank', 'PageRank', 'BiRank', 'SpringRank_AUC', 
                    'PageRank_AUC', 'BiRank_AUC']
        
        writefile = 'ranks_by_year.csv'
        writedir = './'+group+'_data/handmade_scores/'
        
        if writefile in os.listdir(writedir):
            data = pd.pandas.read_csv(writedir+writefile).to_numpy()[:,1:]
            all_data = np.vstack((data, results))
            write_df = pd.DataFrame(all_data, columns=col_list)
            write_df.to_csv(writedir+writefile)
        else:
            write_df = pd.DataFrame([results], columns=col_list)
            write_df.to_csv(writedir+writefile)

        



'''
Get ranks + accuracies by year for each of batters and pitchers for frequency scores.
'''
for year in range(2009,2020):
    for group in ['batter','pitcher']:
        
        filename1='./'+group+'_data/frequency_scores/'+str(year)+'_'+group+'_edges.csv'
        
        print ('Making graph.')
        graph_info = r.makeGraph(filename1, weights=True, val_folds=5)
        G, A, node_list, edge_list, test_edges = graph_info
        
        filename2='./general_data/frequency/'+str(year)+'_edges_only.csv'
        
        print ('Making ranks.')
        spr, pr, bir = r.getRanks(G, A, node_list, filename2, birank_group=group, weights=True)
        sr_raw, pr_raw, br_raw = spr[0], pr[0], bir[0]
        sr_sorted, pr_sorted, br_sorted = spr[1], pr[1], bir[1]
        
        Gacc = nx.DiGraph()
        Gacc.add_edges_from(edge_list[:,:2])
        node_list = list(G.nodes())
        
        Aacc = nx.to_scipy_sparse_matrix(Gacc,
                                      dtype='float64',
                                      nodelist=node_list)
        
        sr_acc, sr_auc = getAccuracy(sr_raw, sr_sorted, Aacc, test_edges)
        pr_acc, pr_auc = getAccuracy(pr_raw, pr_sorted, Aacc, test_edges)
        br_acc, br_auc = getAccuracy(br_raw, br_sorted, Aacc, test_edges)
        results = np.array([year, sr_acc, pr_acc, br_acc, sr_auc, pr_auc, br_auc])
        col_list = ['Year','SpringRank', 'PageRank', 'BiRank', 'SpringRank_AUC', 
                    'PageRank_AUC', 'BiRank_AUC']
        
        writefile = 'ranks_by_year.csv'
        writedir = './'+group+'_data/frequency_scores/'
        
        if writefile in os.listdir(writedir):
            data = pd.pandas.read_csv(writedir+writefile).to_numpy()[:,1:]
            all_data = np.vstack((data, results))
            write_df = pd.DataFrame(all_data, columns=col_list)
            write_df.to_csv(writedir+writefile)
        else:
            write_df = pd.DataFrame([results], columns=col_list)
            write_df.to_csv(writedir+writefile)
        







'''
Get ranks + accuracies by year for each of batters and pitchers for pitch type.
'''
pitch_types=['CH','CU','FC','FF','FS','FT','SI','SL']

for pt in pitch_types:
    for year in range(2009,2020):
        for group in ['batter','pitcher']:
            
                filename1='./'+group+'_data/pitchtype_scores/'+pt+'/'+str(year)+'_'+group+'_edges.csv'
                
                print ('Making graph.')
                graph_info = r.makeGraph(filename1, weights=True, val_folds=5)
                G, A, node_list, edge_list, test_edges = graph_info
                
                filename2='./general_data/pitch_type/'+pt+'/'+str(year)+'_edges_only.csv'
                
                print ('Making ranks.')
                spr, pr, bir = r.getRanks(G, A, node_list, filename2, birank_group=group, weights=True)
                sr_raw, pr_raw, br_raw = spr[0], pr[0], bir[0]
                sr_sorted, pr_sorted, br_sorted = spr[1], pr[1], bir[1]
                
                Gacc = nx.DiGraph()
                Gacc.add_edges_from(edge_list[:,:2])
                node_list = list(G.nodes())
                
                Aacc = nx.to_scipy_sparse_matrix(Gacc,
                                              dtype='float64',
                                              nodelist=node_list)
                
                sr_acc, sr_auc = getAccuracy(sr_raw, sr_sorted, Aacc, test_edges)
                results = np.array([year, sr_acc, sr_auc])
                col_list = ['Year','SpringRank', 'SpringRank_AUC']
                
                writefile = 'ranks_by_year.csv'
                writedir = './'+group+'_data/pitchtype_scores/'+pt+'/'
                
                if writefile in os.listdir(writedir):
                    data = pd.pandas.read_csv(writedir+writefile).to_numpy()[:,1:]
                    all_data = np.vstack((data, results))
                    write_df = pd.DataFrame(all_data, columns=col_list)
                    write_df.to_csv(writedir+writefile)
                else:
                    write_df = pd.DataFrame([results], columns=col_list)
                    write_df.to_csv(writedir+writefile)





'''
Get ranks + accuracies by year for each of batters and pitchers for inning.
'''
innings=list(range(1,10))
for year in range(2009,2020):
    for group in ['batter','pitcher']:
        for inn in innings:
        
            filename1='./'+group+'_data/inning_scores/'+str(inn)+'/'+str(year)+'_'+group+'_edges.csv'
            
            print ('Making graph.')
            graph_info = r.makeGraph(filename1, weights=True, val_folds=5)
            G, A, node_list, edge_list, test_edges = graph_info
            
            filename2='./general_data/inning/'+str(inn)+'/'+str(year)+'_edges_only.csv'
            
            print ('Making ranks.')
            spr, pr, bir = r.getRanks(G, A, node_list, filename2, birank_group=group, weights=True)
            sr_raw, pr_raw, br_raw = spr[0], pr[0], bir[0]
            sr_sorted, pr_sorted, br_sorted = spr[1], pr[1], bir[1]
            
            Gacc = nx.DiGraph()
            Gacc.add_edges_from(edge_list[:,:2])
            node_list = list(G.nodes())
            
            Aacc = nx.to_scipy_sparse_matrix(Gacc,
                                          dtype='float64',
                                          nodelist=node_list)
            
            sr_acc, sr_auc = getAccuracy(sr_raw, sr_sorted, Aacc, test_edges)
            results = np.array([year, sr_acc, sr_auc])
            col_list = ['Year','SpringRank', 'SpringRank_AUC']
            
            writefile = 'ranks_by_year.csv'
            writedir = './'+group+'_data/inning_scores/'+str(inn)+'/'
            
            if writefile in os.listdir(writedir):
                data = pd.pandas.read_csv(writedir+writefile).to_numpy()[:,1:]
                all_data = np.vstack((data, results))
                write_df = pd.DataFrame(all_data, columns=col_list)
                write_df.to_csv(writedir+writefile)
            else:
                write_df = pd.DataFrame([results], columns=col_list)
                write_df.to_csv(writedir+writefile)





