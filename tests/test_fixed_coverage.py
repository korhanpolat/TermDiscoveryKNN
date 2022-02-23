from runexp.pipeline import run_fixed_coverage
import os, sys
import numpy as np
from utils.helper_fncs import load_json

os.chdir(sys.path[0])


alg_type = 'knn'
params = load_json('../config/{}.json'.format(alg_type))
params['expname'] = 'pipeline_test'

feats_dir = '../sample_data/features/phoenix_Signer03_deephand/'

feats_dict = {}
for fname in os.listdir(feats_dir):
    seq_name = fname.replace('.npy','')
    feats_dict[seq_name] = np.load(os.path.join(feats_dir, fname))

covth=10
covmargin=1

matches_df, nodes_df, clusters_list, scores, _ = run_fixed_coverage(feats_dict, params, covth, covmargin)

# print(scores)

if abs(scores['coverageNS']-covth) <= covmargin:
    print('Coverage is within tolerable range')
else:    
    print('Coverage is not within tolerable range')
