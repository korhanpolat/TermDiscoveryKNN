from knn.discoverer import KnnDiscovery
import os, sys
import numpy as np

os.chdir(sys.path[0])
feats_dir = '../sample_data/features/phoenix_Signer03_deephand/'

feats_dict = {}
for fname in os.listdir(feats_dir):
    seq_name = fname.replace('.npy','')
    feats_dict[seq_name] = np.load(os.path.join(feats_dir, fname))
    
params = {'disc': {
                    'a': 3,
                    'dim_fix': 4,
                    'emb_type': 'gauss_kernel',
                    'k': 150,
                    'lmax': 15,
                    'lmin': 2,
                    'metric': 'L2',
                    'norm': False,
                    'olapthr_m': 0.2,
                    'pca': '',
                    'r': 0.21,
                    's': 0.6,
                    'seg_type': 'uniform',
                    'top_delta': 0.05,
                    'use_gpu': True
                    }
        }

knndisc = KnnDiscovery(feats_dict, params)
matches_df = knndisc.run()

print('Found {} matches'.format(len(matches_df)))

print(matches_df.head())

# matches_path = '../sample_data/results/sample_matches.pkl'
# matches_df.save_csv('../sample_data/sample_matches.csv')
# matches_df.to_pickle(matches_path, protocol=3)