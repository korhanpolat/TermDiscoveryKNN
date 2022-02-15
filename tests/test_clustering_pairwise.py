import pandas as pd
from clustering.wrappers import run_clustering
import os, sys

params = {  "clustering": 
                            {
                            "cost_thr": 0.01,
                            "method": "pairwise",
                            "olapthr_m": 0.1
                            },
            "expname" : "clus",
            "exp_root" : os.path.join(sys.path[0] , "..", "sample_data/results")
        }


# load matches
matches_path = os.path.join(params['exp_root'], 'sample_matches.pkl')
matches_df = pd.read_pickle(matches_path)
seq_names = list(set(matches_df['f1']) | set(matches_df['f2']))

nodes_df, clusters_list, postdisc_name = run_clustering(seq_names, matches_df, params)
print('*** post disc completed, found {} segments from {} clusters ***'.format(len(nodes_df), len(clusters_list)))

