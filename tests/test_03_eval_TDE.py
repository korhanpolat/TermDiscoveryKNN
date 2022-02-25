from os.path import join
import os, sys
import pandas as pd

from run.pipeline import evaluate_discovery


SOURCE = "/home/korhan/miniconda3/etc/profile.d/conda.sh"
params = {  "clustering": 
                            {
                            "cost_thr": 0.01,
                            "method": "pairwise",
                            "olapthr_m": 0.1
                            },
            "expname" : "clus",
            "exp_root" : os.path.join(sys.path[0] , "..", "data/sample/results"),
            "eval": {
                    "TDEROOT":"/home/korhan/Desktop/tez/tdev2/tdev2", # change according to your tde build
                    "TDESOURCE":"/home/korhan/miniconda3/etc/profile.d/conda.sh", # to activate conda env from bash script for evaluation, change according to your conda env
                    "njobs":2,
                    "dataset":"phoenix",
                    "config_file": "/home/korhan/Desktop/knn_utd/config/config_phoenix.json",
                    "tderunfile":"/home/korhan/Desktop/knn_utd/run_tde.sh"
                    }
        }



postdisc_name = 'postpairwise_cost{}_olap{}'.format(
                params['clustering']['cost_thr'], params['clustering']['olapthr_m'] )

matches_path = os.path.join(params['exp_root'], 'sample_matches.pkl')
matches_df = pd.read_pickle(matches_path)
seq_names = list(set(matches_df['f1']) | set(matches_df['f2']))


scores = evaluate_discovery(
                params['eval']['tderunfile'], params['eval']['TDEROOT'], params['eval']['TDESOURCE'], 
                join(params['exp_root'], params['expname'], postdisc_name), 
                jobs=params['eval']['njobs'], dataset=params['eval']['dataset'],
                seq_names=seq_names, cnf=params['eval']['config_file']
                )

print('Evaluation complete !! \nscores:')
print(' , '.join(['{}:{}'.format(k,scores[k]) for k in ['ned','coverageNS','grouping_F']]))
