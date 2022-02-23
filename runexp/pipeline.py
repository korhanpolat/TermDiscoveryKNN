import pandas as pd
# from tests.test_eval_TDE import TDEROOT
# from utils.helper_fncs import save_obj, load_obj
from os.path import join
import os
import traceback
import glob

from knn.discoverer import KnnDiscovery
from clustering.wrappers import run_clustering
from utils.helper_fncs import load_json


def get_seq_names_if_exists(matches_path, matches_df):
    seq_lsits = glob.glob(os.path.join(matches_path, '*/seq_names.txt'))
    if len(seq_lsits) > 0: 
        with open(seq_lsits[0],'r') as f:
            seq_names = [x.strip('\n') for x in f.readlines()]
    else:
        seq_names = list(set(matches_df['f1']) | set(matches_df['f2']))

    return seq_names


def run_matches_discovery(feats_dict, params):
    ''' runs the pairwise discovery part, 
        if computed before (i.e. match records are found in exp directory) 
        loads the existing matches from disk
        you can select 2 discovery algorithms
        
        main algorithms
            'sdtw'  :   algorithm of Park&Glass 2008, runs very slowly
            'knn'   :   knn discovery algorithm of Thual, 2018. 

    '''

    # params['expname'] = gen_expname(params)
    matches_path = join(params['exp_root'], params['expname'], 'matches.pkl')

    # run only if the same experiment is not run&saved before
    if not os.path.exists(matches_path):
        os.makedirs(join(params['exp_root'], params['expname']), exist_ok=True)
        seq_names = sorted(feats_dict.keys())

        # if params['disc_method'] == 'sdtw':
        #     matches_info = run_disc_pairwise(feats_dict, params)
        #     matches_df = matches_list_to_df(matches_info)

        if params['disc_method'] == 'knn':
            knndisc = KnnDiscovery(feats_dict, params)
            matches_df = knndisc.run()

        # limit max # of matches
        if len(matches_df) > 200000:
            matches_df = matches_df.sort_values(
                by='cost', ascending=True)[:200000].reset_index(drop=True)

        matches_df.to_pickle(matches_path, protocol=3)

    # load matches if saved file is found
    else: 
        matches_df = pd.read_pickle(matches_path)
        print('*** Matches already discovered !!! ***')
        # try to find seq names 
        seq_names = get_seq_names_if_exists(join(params['exp_root'], params['expname'] ) , matches_df)

    return matches_df, seq_names


def evaluate_discovery(TDEROOT, SOURCE, outdir, jobs=1, dataset='phoenix', seq_names=None, cnf=None):
    import subprocess, sys


    with open(join(outdir, 'seq_names.txt'), 'w') as f: f.write('\n'.join(seq_names))

    # if not os.path.exists(outdir + '/scores.json'):
    os.chdir(sys.path[0])
    cmd = '../run_tde.sh {} {} {} {} {} {} {} {}'.format(TDEROOT, outdir, dataset, 
                                                        'sdtw', outdir + '/scores.json', 
                                                        jobs, cnf, SOURCE )
    subprocess.call(cmd.split())   
    
    try:
        scores = load_json(outdir + '/scores.json')

    except Exception as exc:

        print(traceback.format_exc())
        print(exc)
        scores = {'ned':100.0, 'coverageNS': 0.0}

    if 'ned' not in scores.keys(): scores['ned'] = 100.0
    if 'coverageNS' not in scores.keys(): scores['coverageNS'] = 0.0
    
    return scores
    

def discovery_pipeline(feats_dict, params):
    ''' combine discovery, clustering and evaluation in a single function '''
   
    matches_df, seq_names = run_matches_discovery(feats_dict, params)
    print('*** found {} matches ***'.format(len(matches_df)))

    nodes_df, clusters_list, postdisc_name = run_clustering(seq_names, matches_df, params)
    print('*** post disc completed, found {} segments from {} clusters ***'.format(len(nodes_df), len(clusters_list)))
    
    scores = evaluate_discovery(params['eval']['TDEROOT'], params['eval']['TDESOURCE'], 
                            join(params['exp_root'],params['expname'], postdisc_name), 
                            jobs=params['eval']['njobs'], dataset=params['eval']['dataset'],
                            seq_names=seq_names, cnf=params['eval']['config_file'])

    # _, olapratio = number_of_discovered_frames(nodes_df, clusters_list, returnolap=True)
    # scores['olapratio'] = olapratio
    scores['length_avg'] = (nodes_df.end - nodes_df.start).mean()
    print('*** Coverage: {:.4f}, NED: {:.2f}'.format(scores['coverageNS'], scores['ned']))
    
    return matches_df, nodes_df, clusters_list, scores



def run_fixed_coverage(feats_dict, pars, covth=10, covmargin=1):
    ''' runs the discovery_pipeline() by changing the cost threshold, 
        repeats until coverage is around the desired value (e.g. 10%) 
        params: 
            covth (float):      target coverage percentage, 
                                between 0-100
            covmargin (float):  acceptable tolerance around coverage,
                                wrt 0-100 % units
            patience (int):     max number of runs until coverage is 
                                within tolerance range
    '''
    cov = 0
    matches_df, nodes_df, clusters_list, scores = [None for x in range(4)]

    if covth is None: covth = pars['coverage']['covth'] 

    if 'covmargin' in pars['coverage'].keys(): 
        covmargin = pars['coverage']['covmargin']
    # convert to 1-100 format if covth > 1
    # if covth>1: covmargin *= 100

    if 'patience' in pars['coverage'].keys(): patience = pars['coverage']['patience']
    else:    patience = 10

    cnt = 0

    lr = 0.95 # weight for cost threshold update
    minmatch = 5 # min allowed matches

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
        if (cnt % 5 ==0) : lr = lr * 0.8 # lr decay

    return matches_df, nodes_df, clusters_list, scores, pars


def gen_expname(params):
    ''' generate experiment name using parameters '''

#    if 'basename' not in params.keys(): 
    params['basename'] = '{}_{}_{}'.format(  params['disc_method'], params['CVset'], params['featype'] )

    name = params['basename'] + '_' + '_'.join(['{}{}'.format(k,v) for k,v in params['disc'].items() ])    

    return name


def run_exp(feats_dict, params):
    ''' wraps fixed_coverage experiment, by assigning an experiment name '''

    params['expname'] = gen_expname(params)

    print(params['expname'])
    thresh0 = params['clustering']['cost_thr']
    
    tmp_dict = feats_dict
    
    try:
        matches_df, nodes_df, clusters_list, scores, pars = run_fixed_coverage(
                                                                tmp_dict, params)
    except Exception as exc:

        print(traceback.format_exc())
        print(exc)
        scores = {'ned':100.0, 'coverageNS': 0.0}

    params['clustering']['cost_thr'] = thresh0
    
    return scores, params


# def try_run_exp(feats_dict, params, size = 100, step = 20, genname=False):

#     if genname:
#         params['basename'] = '{}_{}_{}'.format(params['disc_method'], params['CVset'], params['featype'])

#     try:

#         scores = run_exp(feats_dict, params)

#     except Exception as exc:

#         print(traceback.format_exc())
#         print(exc)
#         scores = {'ned':100.0, 'coverageNS': 0.0}

#     return scores

