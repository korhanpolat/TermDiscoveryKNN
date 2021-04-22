import glob
# from utils.ZR_utils import prepare_inputs2, run_disc_2, get_matches_all, change_plebdisc_thr
from os.path import join
import numpy as np
import pickle
import os
import pandas as pd
from utils.pipeline import discovery_pipeline, gen_expname

from utils.paramtune import monitor_callback
import traceback

import time


cols = ['n_clus', 'n_node', 'ned', 'coverage', 'coverageNS', 'coverageNS_f', 'length_avg',
           'grouping_F', 'grouping_P', 'grouping_R', 'token_F', 'token_P', 'token_R', 'type_F', 'type_P', 'type_R',
           'boundary_F', 'boundary_P', 'boundary_R']



def run_until_coverage_th(feats_dict, pars, covth=None, covmargin=0.01):
    ''' repeat the experiment by changing the cost threshold, 
        until coverage is around the desired value (e.g. 10%) 
    '''
    cov = 0
    matches_df, nodes_df, clusters_list, scores = [None for x in range(4)]

    if covth is None: covth = pars['covth'] 

    if 'covmargin' in pars.keys(): covmargin = pars['covmargin']
    if covth>=1: covmargin *= 100

    if 'patience' in pars.keys(): patience = pars['patience']
    else:    patience = 10
    # maxth = pars['maxth'] 
    # mult = pars['mult'] 
    
    cnt = 0
    lr = 0.95
    minmatch = 5    

    cov0 = 0
    th0 = 0
    th = pars['clustering']['cost_thr']

    while abs(cov0-covth) > covmargin:

        if (cnt > patience): break
        # if (th > maxth) : th = maxth

        pars['clustering']['cost_thr'] = th
        print(pars['clustering']['cost_thr'])
        matches_df, nodes_df, clusters_list, scores = discovery_pipeline(feats_dict, pars)
        nmatch = len(matches_df)
    
        cov = scores['coverageNS']
            # monitor_callback(pars, scores, name=pars['csvname'])

        # if (cov > covth) & (cnt == 0): 
        #     pars['clustering']['cost_thr'] = round(pars['clustering']['cost_thr'] * 0.1, 7)
        #     cov = 0
        # else:
        #     pars['clustering']['cost_thr'] = round(pars['clustering']['cost_thr'] * mult, 4)
        

        if (cov == 0) or (cov==cov0):
            cov += 0.0001

        print('trial {} th:{:.5f} cov:{:.5f} err:{:.5f}'.format(cnt, th,cov,cov-covth))
        th_new = th0 + lr*((th-th0)*(covth-cov0))/(cov-cov0)
        cov0=cov; th0=th
        if (th >= 0.99) and (th_new >= 0.99) and (abs(cov0-covth) > covmargin): 
            scores['ned'] = 100.0
            break

        th = max(min(th_new,0.99), minmatch / (nmatch+1))
        th = min(th,0.99)

        cnt += 1
        if (cnt % 5 ==0) : lr = lr * 0.8


    return matches_df, nodes_df, clusters_list, scores, pars



def run_exp(feats_dict, params, size = 100, step = 20):
    ''''''

    params['expname'] = gen_expname(params)

    print(params['expname'])
    thresh0 = params['clustering']['cost_thr']
    
    if 'count' in params :
        count = params['count']
        if ((count // step)+1) > 3: firstn = len(feats_dict)
        else : firstn = ((count // step)+1)*size
        params['firstn'] = firstn
        print('Running first {} files, for {}th experiment'.format(firstn, count))

        tmp_dict = {k: feats_dict[k] for k in sorted(feats_dict.keys())[:firstn]}
    else:
        tmp_dict = feats_dict
    
    try:
        matches_df, nodes_df, clusters_list, scores, pars = run_until_coverage_th(
                                                                tmp_dict, params)
    except Exception as exc:

        print(traceback.format_exc())
        print(exc)
        scores = [0]

    params['clustering']['cost_thr'] = thresh0
    
    return {**scores, **params['disc']}



def try_run_exp(feats_dict, params, size = 100, step = 20, genname=False):

    # if 'basename' not in params.keys(): 
    if genname:
        params['basename'] = '{}_{}_{}'.format(params['disc_method'], params['CVset'], params['featype'])
    # if 'csvname' not in params.keys(): params['csvname'] = params['basename0']

    try:

        scores = run_exp(feats_dict, params)

    except Exception as exc:

        print(traceback.format_exc())
        print(exc)
        scores = {'ned':100.0, 'coverageNS': 0.0}

    return scores


def pretty_results_single(scores, params):

    tmp = pd.DataFrame(evals)
    res_dict = {**scores[cols],
                **params['disc'],
                **tmp.mean().reindex(cols), 
                **dict( (k+'_std',v) for k,v in tmp.std().reindex(cols).items() ),
                }

    return res_dict



def pretty_results(evals, params):

    tmp = pd.DataFrame(evals)
    res_dict = {**{'n_exp': len(evals)},
                **params['disc'],
                **tmp.mean().reindex(cols), 
                **dict( (k+'_std',v) for k,v in tmp.std().reindex(cols).items() ),
                }

    return res_dict


def cv_experiment(seq_names, feats_dict, params, nfold = 5):

    foldsize = int(len(feats_dict) / nfold)
    evals = []
    params['basename0'] = '{}_{}_{}'.format(params['disc_method'], params['CVset'], params['featype'])
    # if 'csvname' not in params.keys(): params['csvname'] = params['basename0']

    tstart = time.time()

    for k in range(nfold):
                
        s, e = k*foldsize, (k+1)*foldsize
        # tmp_dict = {k: feats_dict[k] for k in seq_names[s:e]}
        # params['basename'] = params['basename0'] + '_{}_{}'.format(s,e)

        tmp_dict = {key: feats_dict[key] for key in seq_names[k::nfold]}
        params['basename'] = params['basename0'] + '_fold{}-{}'.format(k+1, nfold)

        # if params['disc_method'] == 'zr_cat': params['uniq_id'] = '{}_{}'.format(k,nfold)
        print(params['basename'])


        scores = try_run_exp(tmp_dict , params)
        
        if type(scores) is not list : evals.append(scores)
            
    print('=== {:.2f}s elapsed for {} fold CV run ==='.format(time.time()-tstart, nfold))
    print('=== average score {} for {} experiments ==='.format(np.mean([sc['ned'] for sc in evals]), len(evals)))

    # return pd.DataFrame(evals).mean(), pd.DataFrame(evals).var()
    return pretty_results(evals, params)



def grid_exp(feats_dict, params, grid_dict):

    evals = []
    grid_keys = sorted(grid_dict.keys())
    
    if len(grid_keys)>0:
        k = 0
        for params['disc'][grid_keys[k]] in grid_dict[grid_keys[k]]:
            if len(grid_keys)>1:
                l = 1
                for params['disc'][grid_keys[l]] in grid_dict[grid_keys[l]]:
                    if len(grid_keys)>2:
                        m = 2
                        for params['disc'][grid_keys[m]] in grid_dict[grid_keys[m]]:
                            if len(grid_keys)>3:
                                n = 3
                                for params['disc'][grid_keys[n]] in grid_dict[grid_keys[n]]:
                                    scores = try_run_exp(feats_dict, params)
                                    evals.append(scores)
                            else:
                                scores = try_run_exp(feats_dict, params)
                                evals.append(scores)
                    else:
                        scores = try_run_exp(feats_dict, params)
                        evals.append(scores)
            else:
                scores = try_run_exp(feats_dict, params)
                evals.append(scores)
    
    return evals