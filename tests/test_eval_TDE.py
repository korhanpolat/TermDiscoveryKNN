from os.path import join
import os, sys
import pandas as pd

from runexp.pipeline import evaluate_discovery


SOURCE = "/home/korhan/miniconda3/etc/profile.d/conda.sh"
params = {  "clustering": 
                            {
                            "cost_thr": 0.01,
                            "method": "pairwise",
                            "olapthr_m": 0.1
                            },
            "expname" : "clus",
            "exp_root" : os.path.join(sys.path[0] , "..", "sample_data/results"),
            "eval": {
                    "TDEROOT":"/home/korhan/Desktop/tez/tdev2/tdev2", # change according to your tde build
                    "TDESOURCE":"/home/korhan/miniconda3/etc/profile.d/conda.sh", # to activate conda env from bash script for evaluation, change according to your conda env
                    "njobs":2,
                    "dataset":"phoenix",
                    "config_file": "/home/korhan/Desktop/knn_utd/config/config_phoenix.json",
                    }
        }



postdisc_name = 'postpairwise_cost{}_olap{}'.format(
                params['clustering']['cost_thr'], params['clustering']['olapthr_m'] )

matches_path = os.path.join(params['exp_root'], 'sample_matches.pkl')
matches_df = pd.read_pickle(matches_path)
seq_names = list(set(matches_df['f1']) | set(matches_df['f2']))


scores = evaluate_discovery(params['eval']['TDEROOT'], params['eval']['TDESOURCE'], 
                    join(params['exp_root'],params['expname'], postdisc_name), 
                    jobs=params['eval']['njobs'], dataset=params['eval']['dataset'],
                    seq_names=seq_names, cnf=params['eval']['config_file'])

