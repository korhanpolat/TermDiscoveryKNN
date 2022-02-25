from run.pipeline import discovery_pipeline
import os, sys
import numpy as np
from utils.helper_fncs import load_json

os.chdir(sys.path[0])


alg_type = 'knn'
params = load_json('../config/{}.json'.format(alg_type))
params['expname'] = 'pipeline_test'

feats_dir = '../data/sample/features/phoenix_Signer03_deephand/'

feats_dict = {}
for fname in os.listdir(feats_dir):
    seq_name = fname.replace('.npy','')
    feats_dict[seq_name] = np.load(os.path.join(feats_dir, fname))



matches_df, nodes_df, clusters_list, scores = discovery_pipeline(feats_dict, params)

print('Discovery completed !! \nscores:')
print(' , '.join(['{}:{}'.format(k,scores[k]) for k in ['ned','coverageNS','grouping_F']]))
