from utils.db_utils import get_seq_names
from run.pipeline import run_exp


cvsets = set(['A','B','C'])
signers_per_set = {'A': [4,8,9], 'B':[2,5,7], 'C':[1,3,6]}



def run_for_set(cvset, feat_loader, params, signer_independent=False):
    # get scores for devset_x (params_optim)
    if signer_independent:
        signers = ['Signer0{}'.format(x) for x in signers_per_set[cvset]]
    else:
        signers = []
    
    scores_dict = dict()
    for params['CVset'] in [cvset] + signers:
        seq_names = get_seq_names(params)
        
        feats_dict = feat_loader(seq_names, params)
        
        tmp_feats = {key: feats_dict[key] for key in seq_names}
        scores, params = run_exp(tmp_feats, params, genname=True)
        while scores['ned'] > 99: 
            params['disc']['top_delta'] += 0.01
            scores, params = run_exp(tmp_feats, params, genname=True)

                
        scores_dict[params['CVset']] = {**scores, **params['disc'], **params['clustering']}
        params['disc']['top_delta'] = 0.05

    return scores_dict



