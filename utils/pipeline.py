
from utils.eval import evaluate

from scipy.signal import medfilt

from joblib import Parallel, delayed
from itertools import combinations
from utils.clustering import cluster_adj_mat, remove_single_nodes, find_single_nodes_in_matrix, make_symmetric


from scipy.signal import triang, convolve, find_peaks
from utils.sdtw_funcs import sdtw_jit, LCMA_jit, joints_loss, sdtw_np, LCMA_jit_new

from utils.ZR_utils import new_match_dict, change_post_disc_thr, post_disc2, get_nodes_df, get_clusters_list, run_disc_ZR, get_matches_all
from utils.ZR_cat import run_disc_ZR_feats_concat
from utils.knn.discoverer import KnnDiscovery


import pandas as pd
import numpy as np
from utils.feature_utils import get_features_array_for_seq_name, get_paths_for_features, normalize_frames, kl_symmetric_pairwise
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity
from utils.eval import evaluate, save_token_frames_per_cluster, number_of_discovered_frames
from utils.helper_fncs import save_obj, load_obj
from os.path import join
import os
import traceback
from utils.discovery_utils import *

from numba import jit, prange
# from sdtw_pipeline import run_disc_pairwise

#from utils.knn.discoverer import KnnDiscovery

# change according to your tde build
TDEROOT='/home/korhan/Desktop/tez/tdev2/tdev2'    
# to activate conda env from bash script for evaluation, change according to your conda env
SOURCE = glob.glob('/home/korhan/*/etc/profile.d/conda.sh')[0] 

def gen_expname(params):

#    if 'basename' not in params.keys(): 
    params['basename'] = '{}_{}_{}'.format(  params['disc_method'], params['CVset'], params['featype'] )

    name = params['basename'] + '_' + '_'.join(['{}{}'.format(k,v) for k,v in params['disc'].items() ])    
    return name




def matches_list_to_df(matches_info):
    matches_df = pd.DataFrame.from_records(matches_info,
                                           columns=['f1', 'f2', 'f1_start', 'f1_end', 'f2_start', 'f2_end', 'score']
                                           )

    matches_df = matches_df.astype(dtype={'f1': str, 'f2': str,
                                          'f1_start': int, 'f1_end': int, 'f2_start': int, 'f2_end': int,
                                          'score': float})  # type: pd.DataFrame

    matches_df.columns = ['f1', 'f2', 'f1_start', 'f1_end', 'f2_start', 'f2_end', 'cost']

    return matches_df


def get_seq_names_if_exists(matches_path, matches_df):
    seq_lsits = glob.glob(os.path.join(matches_path, '*/seq_names.txt'))
    if len(seq_lsits) > 0: 
        with open(seq_lsits[0],'r') as f:
            seq_names = [x.strip('\n') for x in f.readlines()]
    else:
        seq_names = list(set(matches_df['f1']) | set(matches_df['f2']))

    return seq_names


def select_matches_length_ratio(matches_df, a=0.75):

    tmp_matches = matches_df.copy()
    tmp_matches['Lf1'] = tmp_matches['f1_end'] - tmp_matches['f1_start']
    tmp_matches['Lf2'] = tmp_matches['f2_end'] - tmp_matches['f2_start']

    return tmp_matches.loc[(tmp_matches['Lf2']/tmp_matches['Lf1'] > a) \
                           & (tmp_matches['Lf1']/tmp_matches['Lf2'] > a)]


def fuse_zr_knn(feats_dict, params_fuse):

    params = params_fuse['zr']
    matches_path = join(params['exp_root'], params['expname'], 'matches.pkl')
    if not os.path.exists(matches_path):
        matches_zr = run_disc_ZR_feats_concat(feats_dict, params)
    else: 
        matches_zr = pd.read_pickle(matches_path)
        print('*** Matches already discovered !!! ***')
        # try to find seq names 
        seq_names_zr = get_seq_names_if_exists(join(params['exp_root'], params['expname'] ) , matches_zr)

    matches_zr_raw = get_matches_all(join(params['exp_root'], params['expname'], 'matches/') )
    matches_zr_raw['cost'] = 1 - matches_zr_raw['score']
    stats = matches_zr_raw['cost'].mean(), matches_zr_raw['cost'].std()
    matches_zr['cost_norm'] = (matches_zr['cost'] - stats[0] ) / stats[1]


    params = params_fuse['knn']
    matches_path = join(params['exp_root'], params['expname'], 'matches.pkl')
    if not os.path.exists(matches_path):
        knndisc = KnnDiscovery(feats_dict, params)
        matches_knn = knndisc.run()
    else: 
        matches_knn = pd.read_pickle(matches_path)
        print('*** Matches already discovered !!! ***')
        # try to find seq names 
        seq_names_knn = get_seq_names_if_exists(join(params['exp_root'], params['expname'] ) , matches_knn)


    stats = matches_knn['cost'].mean(), matches_knn['cost'].std()
    matches_knn['cost_norm'] = (matches_knn['cost'] - stats[0] ) / stats[1]

    matches_knn.cost = matches_knn.cost_norm + params_fuse['disc']['shift']
    matches_zr.cost = (matches_zr.cost_norm + params_fuse['disc']['shift']) * params_fuse['disc']['scale_param']
    matches_fuse = pd.concat([matches_zr, matches_knn])

    matches_fuse = select_matches_length_ratio(matches_fuse, a=params_fuse['disc']['a'])

    expdir = os.path.join(params_fuse['exp_root'], params_fuse['expname'])
    os.makedirs(expdir,exist_ok=True)

    seq_names = list(set(seq_names_zr) | set(seq_names_knn) )


    return matches_fuse, seq_names



def run_matches_discovery(feats_dict, params):
    ''' runs the pairwise discovery part, 
        if computed before (i.e. match records are found in exp directory) 
        loads the existing matches from disk
        you can select 3 discovery algorithms, and their variations as explained below
        
        main algorithms
            'sdtw'  :   algorithm of Park&Glass 2008, runs very slowly
            'zr'    :   efficient sdtw algorithm of Jansen,2011, a.k.a. ZR Tools
            'knn'   :   knn discovery algorithm of Thual, 2018. 
        variations
            'fuse'  :   fusion of ZRTools and KNN based discovery (experimental)
            'zr_cat':   concatenates the feature arrays, runs ZR discovery and returns the 
                        original time indices back. Faster than ZRTools because diagonal 
                        line segment search is performed in bigger but fewer matrices

    '''

    if params['disc_method'] == 'fuse':
        matches_df, seq_names = fuse_zr_knn(feats_dict, params)
        return matches_df, seq_names

    # params['expname'] = gen_expname(params)
    matches_path = join(params['exp_root'], params['expname'], 'matches.pkl')

    if not os.path.exists(matches_path):
        os.makedirs(join(params['exp_root'], params['expname']), exist_ok=True)
        seq_names = sorted(feats_dict.keys())

        if params['disc_method'] == 'sdtw':
            matches_info = run_disc_pairwise(feats_dict, params)
            matches_df = matches_list_to_df(matches_info)

        if params['disc_method'] == 'zr':
            matches_df = run_disc_ZR(feats_dict, params)

        if params['disc_method'] == 'zr_cat':
            matches_df = run_disc_ZR_feats_concat(feats_dict, params)

        if params['disc_method'] == 'knn':
            knndisc = KnnDiscovery(feats_dict, params)
            matches_df = knndisc.run()


        if len(matches_df) > 200000:
            matches_df = matches_df.sort_values(by='cost', ascending=True)[:200000].reset_index(drop=True)

        matches_df.to_pickle(matches_path, protocol=3)

    else: 
        matches_df = pd.read_pickle(matches_path)
        print('*** Matches already discovered !!! ***')
        # try to find seq names 
        seq_names = get_seq_names_if_exists(join(params['exp_root'], params['expname'] ) , matches_df)

    return matches_df, seq_names


def apply_cost_thr(matches_df, cost_thr=0.1):
    cost_val = matches_df.cost.sort_values(ascending=True).values[round(len(matches_df) * cost_thr)]
    matches_df = matches_df.loc[matches_df['cost'] < cost_val].copy()
    matches_df.reset_index(inplace=True, drop=True)
    matches_df['edge_w'] = matches_df['cost'].apply(lambda x: (cost_val - x) / cost_val)

    return matches_df


def apply_cost_edge_w(matches_df, cost_thr=0.1):
    # apply threshold to normalized edge weights
    cost_max = matches_df.cost.max()
    matches_df['edge_w'] = matches_df['cost'].apply(lambda x: (cost_max - x) / cost_max)
    
    cost_val = matches_df['edge_w'].sort_values(ascending=False).values[round(len(matches_df) * cost_thr)]
    matches_df = matches_df.loc[matches_df['edge_w'] > cost_val].copy()
    matches_df.reset_index(inplace=True, drop=True)

    return matches_df


def deduplicate_matches(matches_df, params_clus):
    # apply cost threshold
    matches_df = apply_cost_edge_w(matches_df, params_clus['cost_thr'])
    matches_df = matches_df.sort_values(by='cost').reset_index(drop=True)
    # convert to arrays for speed
    fnames, f1f2arr, s1e1s2e2array, wgtharray = matches_to_arrays(matches_df, cols=['cost'])
    # find indices to remove
    to_remove = find_pairwise_overlaps_NMS(f1f2arr, s1e1s2e2array, wgtharray, params_clus['olapthr_m'])
    # find same file overlaps too
    to_remove[find_same_match_overlaps(f1f2arr, s1e1s2e2array, wgtharray, params_clus['olapthr_m'])] = True
    # clean the df
    matches_df.drop(np.nonzero(to_remove)[0], inplace=True)
    
    return matches_df.reset_index()


def compute_similarity_profile(matches_df, seq_names):
    # find the similarity profile, updated

    max_frame = matches_df[['f1_end','f2_end']].max().max() + 1
    similarity_profile = dict()

    for i, seq_name in enumerate(sorted(seq_names)):
        similarity_profile[seq_name] = np.zeros(max_frame)

    for i, seq_name in enumerate(sorted(seq_names)):

        for j in range(1,3):

            s = 'f' + str(j)+'_start'
            e = 'f' + str(j)+'_end'

            tmp = matches_df[[s,e]][ matches_df['f' + str(j) ] == seq_name ]
            ew = 'edge_w'

            tmp = matches_df[[s, e, ew]][ matches_df['f' + str(j) ] == seq_name ]

            for ri, row in tmp.iterrows():
                similarity_profile[seq_name][int(row[s]):int(row[e])] += row[ew]

    return similarity_profile



def find_node_centers(similarity_profile, peak_thr=1, filter_w=13):

    node_centers = dict()

    triang_w = triang(filter_w) / (filter_w/2)

    for name, vec in similarity_profile.items():

        filtered_vec = convolve(vec, triang_w, mode='same' )
        peaks = find_peaks(filtered_vec, height=peak_thr)
        node_centers[name] = peaks[0]

    return node_centers



def nodes_list_to_df(node_centers):
    # expand note centers to a list
    nodes_centers_list = [] #pd.DataFrame(columns=['name','idx'])
    for name, vec in node_centers.items():
        for i in vec:
            nodes_centers_list.append({'filename':name, 'idx':i})

    nodes_centers_df = pd.DataFrame.from_records(nodes_centers_list, columns=['filename','idx'])

    return nodes_centers_df



@jit
def find_included_node_centers_fast(node_centers_array,f1,f2, s1,e1,s2,e2):
      
    nidx1 = np.where([(node_centers_array[:,0] == f1) & (node_centers_array[:,1]>s1)& (node_centers_array[:,1]<e1) ])[1]
    nidx2 = np.where([(node_centers_array[:,0] == f2) & (node_centers_array[:,1]>s2)& (node_centers_array[:,1]<e2) ])[1]
    
    return nidx1, nidx2


@jit
def compute_adjacency_matrix_fast_engine(n, node_centers_array, seqnameidx, s1e1s2e2, weights ):
    
    edges_mat = np.zeros((n, n))
    path_idx_cnt = np.zeros((n, n), dtype=np.int32)
    path_idx_mat = np.zeros((n, n, 40), dtype=np.int32)  # [idx1,idx2,cnt] -> row i


    for i,  edge_w in enumerate(weights):
        
        f1,f2 = seqnameidx[i]
        s1,e1,s2,e2 = s1e1s2e2[i]
        
        idx1, idx2 = find_included_node_centers_fast(node_centers_array, f1,f2, s1,e1,s2,e2)

        if len(idx1) == 0 or len(idx2) == 0: continue

        edges_mat[np.ix_(idx1,idx2)] += edge_w

        path_idx_mat[np.ix_(idx1,idx2, path_idx_cnt[np.ix_(idx1,idx2)].reshape(-1))] = i
        path_idx_mat[np.ix_(idx2,idx1, path_idx_cnt[np.ix_(idx2,idx1)].reshape(-1))] = i

        path_idx_cnt[np.ix_(idx1,idx2)] += 1
        path_idx_cnt[np.ix_(idx2,idx1)] += 1

        
    return edges_mat, path_idx_mat, path_idx_cnt



def compute_adjacency_matrix_fast(seq_names, nodes_centers_df, matches_df):

    n = len(nodes_centers_df)
    
    seq_names = sorted(seq_names)

    # convert node centers to array
    filesidx = nodes_centers_df.filename.apply(lambda x: seq_names.index(x)).values
    centers = nodes_centers_df.idx.values
    node_centers_array = np.vstack([filesidx,centers]).T # shape nnode x 2

    # convert matches to arrays too
    seq1nameidx = matches_df.f1.apply(lambda x: seq_names.index(x)).values
    seq2nameidx = matches_df.f2.apply(lambda x: seq_names.index(x)).values
    seqnameidx = np.vstack([seq1nameidx, seq2nameidx]).T

    s1e1s2e2 = matches_df[['f1_start','f1_end','f2_start','f2_end']].values
    weights = matches_df['edge_w'].values

    edges_mat, path_idx_mat, path_idx_cnt = compute_adjacency_matrix_fast_engine(
        n, node_centers_array, seqnameidx, s1e1s2e2, weights )

    return edges_mat, path_idx_mat, path_idx_cnt



def remove_isolated_nodes(edges_mat, path_idx_mat, path_idx_cnt, nodes_centers_df):

    nodes_to_delete = find_single_nodes_in_matrix(edges_mat)

    if len(nodes_to_delete) > 0:
#        print('{} nodes are deleted..'.format(len(nodes_to_delete)))
        edges_mat = remove_single_nodes(edges_mat, nodes_to_delete)
        path_idx_mat = remove_single_nodes(path_idx_mat, nodes_to_delete)
        path_idx_cnt = remove_single_nodes(path_idx_cnt, nodes_to_delete)

        # also need to delete those nodes from node centers df
        nodes_centers_df.drop(nodes_to_delete, inplace=True)
        nodes_centers_df.reset_index(inplace=True, drop=True)

    return edges_mat, path_idx_mat, path_idx_cnt, nodes_centers_df



def determine_node_intervals(matches_df, nodes_centers_df, clusters_list, path_idx_mat, path_idx_cnt, included_nodes):
    # find intervals of nodes that share a common path and are in the same cluster

    intervals = np.zeros((len(nodes_centers_df), 3))

    for clus in clusters_list:

        for pair in combinations(clus, 2):  # same cluster

            n_center_for_pair = path_idx_cnt[pair]

            #        if n_center_for_pair > 0:  # if path exists between the two nodes

            for k in range(n_center_for_pair):

                row = matches_df.loc[path_idx_mat[pair][k]]

                if row['f1'] == nodes_centers_df.filename[pair[0]] and row['f2'] == nodes_centers_df.filename[pair[1]]:
                    ix1, ix2 = pair
                elif row['f2'] == nodes_centers_df.filename[pair[0]] and row['f1'] == nodes_centers_df.filename[pair[1]]:
                    ix2, ix1 = pair
                # else:
                    # pdb.set_trace()

                    # raise Exception('Filenames of pair does not match!!')

                intervals[ix1, 0] += row['f1_start']
                intervals[ix1, 1] += row['f1_end']
                intervals[ix1, 2] += 1
                intervals[ix2, 0] += row['f2_start']
                intervals[ix2, 1] += row['f2_end']
                intervals[ix2, 2] += 1

    intervals = intervals[included_nodes]
    start_indices = np.int64(intervals[:, 0] / intervals[:, 2])
    end_indices = np.int64(intervals[:, 1] / intervals[:, 2])

    return start_indices, end_indices


def get_segment_info(matches_df, center):
    tmp_df = matches_df.loc[(matches_df['f1'] == center[0].filename) & 
               (matches_df['f1_start'] <= center[0].idx) & 
               (matches_df['f1_end'] >= center[0].idx) &
               (matches_df['f2'] == center[1].filename) & 
               (matches_df['f2_start'] <= center[1].idx) & 
               (matches_df['f2_end'] >= center[1].idx)]

    return tmp_df


def update_interval(intervals, ix, val ):
    s, e, w = val   
    
    intervals[ix,0] += s * w
    intervals[ix,1] += e * w
    intervals[ix,2] += w
    

def update_interval_pair(intervals, pair, row):
    
    ix1, ix2 = pair

    update_interval(intervals, ix1, row[['f1_start','f1_end','edge_w']].values)
    update_interval(intervals, ix2, row[['f2_start','f2_end','edge_w']].values)
    

def determine_node_intervals_new(matches_df, nodes_centers_df, clusters_list, included_nodes):
    
    intervals = np.zeros((len(nodes_centers_df), 3) )

    for clus in clusters_list:

        for pair in combinations(clus, 2):  # same cluster
            pair = list(pair)

            center = [nodes_centers_df.loc[p] for p in pair]
            segs = get_segment_info(matches_df, center)

            if len(segs) == 0: 
                center.reverse()
                pair.reverse()
                segs = get_segment_info(matches_df, center)

            for s, row in segs.iterrows():
                update_interval_pair(intervals, pair, row)

    assert list(np.where(np.any(intervals, axis=1))[0]) == included_nodes 

    intervals = intervals[included_nodes]
    start_indices = np.uint64(intervals[:, 0] / intervals[:, 2])
    end_indices = np.uint64(intervals[:, 1] / intervals[:, 2])
    
    return start_indices, end_indices



def update_nodes_and_clusters(nodes_centers_df, clusters_list, included_nodes, excluded_nodes):
    # update nodes and cluster indices: remove nodes that dont belong to a community
    # nodes_centers_df.drop(excluded_nodes, inplace=True)
    nodes_centers_df = nodes_centers_df.loc[included_nodes]
    nodes_centers_df.reset_index(drop=True, inplace=True)

    for i, cluster in enumerate(clusters_list):
        for j, c in enumerate(cluster):
            clusters_list[i][j] = included_nodes.index(c)

    return nodes_centers_df, clusters_list



@jit
def get_vals(i,j, f1f2arr, s1e1s2e2array, wgtharray):
    idxmatch = np.where((f1f2arr[:,0]==i) & (f1f2arr[:,1]==j))
    return s1e1s2e2array[idxmatch], np.round(wgtharray[idxmatch],3)


def matches_to_arrays(matches_thd, cols):
    # convert df to arrays for faster search
    # cols: names of weight columns
    fnames = sorted(list(set(matches_thd.f1.unique()) | set(matches_thd.f2.unique())))

    f1f2arr = np.column_stack([matches_thd[fs].apply(lambda x: fnames.index(x)) for fs in ['f1','f2']])
    s1e1s2e2array = matches_thd[['f1_start','f1_end' ,'f2_start','f2_end']].values
    wgtharray = matches_thd[cols].values

    return fnames, np.uint64(f1f2arr), np.uint64(s1e1s2e2array), np.float64(wgtharray)


def write_matches_ZR(matches_thd, postdisc_path):
    ''' write to txt as zr format
        for each file pair, list matching fragments as new lines
        row fmt : s1 e1 s2 e2 ew ~rho
        matches_thd: thresholded df    '''

    cols = ['edge_w', 'cost']
    fnames, f1f2arr, s1e1s2e2array, wgtharray = matches_to_arrays(matches_thd, cols)
    n = len(fnames)

    # prepare directories and necessary files
    os.makedirs(join(postdisc_path,'matches'), exist_ok=True)
    os.makedirs(join(postdisc_path,'results'), exist_ok=True)
    matchtxt = join(postdisc_path,'matches','out.1')
    with open(join(postdisc_path,'files.base'), 'w') as f: f.write('\n'.join(fnames))
    
    lines = []
    for i in range(n):
        for j in range(i+1,n):
            f1,f2 = fnames[i], fnames[j]
            lines.append(' '.join([f1,f2]) + '\n')
            idxs, weights = get_vals(i,j, f1f2arr, s1e1s2e2array, wgtharray)
            string = '\n'.join([ ' '.join(
                [str(x) for x in idxs[k] ] + [str(x) for x in weights[k] ] ) for k in range(len(idxs))])
            if len(string)>0: lines.append(string + '\n')
    
    # write to txt
    with open(matchtxt,'w') as f: f.writelines(lines)
        



def post_disc(seq_names, matches_df, params):
    ''' modularity based clustering, as described in 
        Park&Glass, "Unsupervised pattern discovery in speech", 2008 '''

    matches_df = apply_cost_edge_w(matches_df, params['cost_thr'])
    print('*** mean edge weight is {:.3f} after cost threshold ***'.format(matches_df['edge_w'].mean()))

    similarity_profile = compute_similarity_profile(matches_df, seq_names )
    print('*** similarity profile computed ***')

    node_centers = find_node_centers(similarity_profile, peak_thr=params['peak_thr'], filter_w=13)
    print('*** node centers found ***')

    nodes_df = nodes_list_to_df(node_centers)

    edges_mat, path_idx_mat, path_idx_cnt = compute_adjacency_matrix_fast(seq_names, nodes_df, matches_df.sort_index())
    print('*** graph constructed ***')

    # edges_mat = make_symmetric(edges_mat)

    # edges_mat, path_idx_mat, path_idx_cnt, nodes_df = remove_isolated_nodes(
                            # edges_mat, path_idx_mat, path_idx_cnt, nodes_df)

    clusters_list_tmp, memberships, modularity = cluster_adj_mat(
                            edges_mat, q_thr = params['modularity_thr'], method=params['clus_alg'])
    print('*** graph clustering done !! ***')

    clusters_list, excluded_clusters, included_nodes, excluded_nodes = prune_clusters(
                            clusters_list_tmp, cluster_thr=params['min_cluster_size'])

    # start_indices, end_indices = determine_node_intervals(
                            # matches_df, nodes_df, clusters_list, path_idx_mat, path_idx_cnt, included_nodes )
    start_indices, end_indices = determine_node_intervals_new(matches_df, nodes_df, clusters_list, included_nodes)                            
    print('*** intervals found ***')

    nodes_df, clusters_list = update_nodes_and_clusters(nodes_df, clusters_list, included_nodes, excluded_nodes)

    nodes_df['start'] = start_indices
    nodes_df['end'] = end_indices

    # deduplicate clusters
    clusters_list = dedup_clusters(clusters_list, nodes_df, params['dedupthr'], params['min_cluster_size'])

    return nodes_df, clusters_list


def nodes_starts_from1(nodes_df, clusters_list):
    # ensure the incides start from at least 1
    nodes_df.index += 1
    for c,clus in enumerate(clusters_list):
        clusters_list[c] = list(np.array(clus)+1)

    return nodes_df, clusters_list


def run_clustering_Modularity(seq_names, matches_df, params):
    
    nodes_df, clusters_list = post_disc(seq_names, matches_df, params)

    nodes_df, clusters_list = nodes_starts_from1(nodes_df, clusters_list)

    return nodes_df, clusters_list


def run_clustering_ZR(matches_df, params, postdisc_path, postdisc_name):
    
    prclus = params['clustering']
    # apply cost thr
    matches_thd = apply_cost_edge_w(matches_df.copy(), prclus['cost_thr'])
    del matches_df
    # convert df to txt 
    write_matches_ZR(matches_thd, postdisc_path)
    # call zr post disc
    change_post_disc_thr(olapthr=prclus['olapthr'], dedupthr=prclus['dedupthr'], 
                         durthr=5, rhothr=1000, min_edge_w=prclus['min_ew'])
    respath = post_disc2(params['exp_root'],join(params['expname'],postdisc_name), prclus['dtwth'])
    # read nodes and clusters
    nodes_df = get_nodes_df(respath)
    clusters_list = get_clusters_list(respath)

    return nodes_df, clusters_list


def match_pairs_as_clusters(matches_df):
    # convert to nodes and clusters
    nodes_list = []
    clusters_list = []

    for i,row in matches_df.iterrows():
        nodes_list.append((row['f1'], row['f1_start'], row['f1_end']))
        nodes_list.append((row['f2'], row['f2_start'], row['f2_end']))

        clusters_list.append([2*i,2*i+1])

    nodes_df = pd.DataFrame(nodes_list, columns=['filename','start','end'] )
    
    return nodes_df, clusters_list


def run_clustering_pairs(matches_df, params_clus):
    print('*** pairwise clustering ***')
    # if not 'disc' in params_clus.keys():
    matches_df = deduplicate_matches(matches_df, params_clus)

    # TO DO: bu ikisini ayir
    nodes_df, clusters_list = match_pairs_as_clusters(matches_df)

    nodes_df, clusters_list = nodes_starts_from1(nodes_df, clusters_list)

    return nodes_df, clusters_list


def run_custom_clustering(matches_df, params):

    if not 'f1_id' in matches_df.columns:
        seq_names = sorted(set(matches_df.f1) | set(matches_df.f2))

        matches_df['f1_id'] = matches_df.f1.apply(lambda x: seq_names.index(x))
        matches_df['f2_id'] = matches_df.f2.apply(lambda x: seq_names.index(x))

    # apply cost threshold
    matches_df = apply_cost_edge_w(matches_df, params['cost_thr'])
    matches_df = matches_df.sort_values(by='cost').reset_index(drop=True)    

    edges_mat = matches_to_edges_mat(matches_df, params['mix_ratio'])

    clusters_list_tmp, memberships, modularity = cluster_adj_mat(
                            edges_mat, q_thr = params['modularity_thr'], method=params['clus_alg'])

    nodes_df = matches_to_nodes(matches_df)

    clusters_list = dedup_clusters(clusters_list_tmp, nodes_df, params['dedupthr'], params['min_cluster_size'])
                
    nodes_df, clusters_list = nodes_starts_from1(nodes_df, clusters_list)
    
    return nodes_df, clusters_list


def gen_postdisc_name(params):
    if params['method'] == 'modularity':
        postdisc_name = 'post_cost{}_peak{}_q{}_{}Alg_mc{}'.format(
                params['cost_thr'], params['peak_thr'], params['modularity_thr'],
                params['clus_alg'], params['min_cluster_size'])

    if params['method'] == 'zr17':
        postdisc_name = 'postZR_cost{}_olap{}_dedup{}_edw{}_dtw{}'.format(
                params['cost_thr'], params['olapthr'], 
                params['dedupthr'], params['min_ew'], params['dtwth'])

    if params['method'] == 'custom':
        postdisc_name = 'post_customclus_cost{}_{}Alg_dedup{}_mix{}'.format(
                params['cost_thr'], params['clus_alg'], 
                params['dedupthr'], params['mix_ratio'])

    if params['method'] == 'pairwise':
        postdisc_name = 'postpairwise_cost{}_olap{}'.format(
                params['cost_thr'], params['olapthr_m'] )

    return postdisc_name


def run_clustering(seq_names, matches_df, params):
    # runs the clustering part, if not computed before
       
    postdisc_name = gen_postdisc_name(params['clustering'])
    
    postdisc_path = join(params['exp_root'], params['expname'], postdisc_name)

    if (os.path.exists(join(postdisc_path,'nodes.pkl')) and      
        os.path.exists(join(postdisc_path,'clusters.pkl'))):
         
        nodes_df, clusters_list = pickle_load_nodes_clusters(postdisc_path)

    else:

        if params['clustering']['method'] == 'modularity':

            os.makedirs(postdisc_path, exist_ok=True)
            nodes_df, clusters_list = run_clustering_Modularity(
                seq_names, matches_df, params['clustering'])

        if params['clustering']['method'] == 'zr17':

            nodes_df, clusters_list = run_clustering_ZR(
                matches_df, params, postdisc_path, postdisc_name)

        if params['clustering']['method'] == 'custom':

            os.makedirs(postdisc_path, exist_ok=True)
            nodes_df, clusters_list = run_custom_clustering(
                matches_df, params['clustering'])

        if params['clustering']['method'] == 'pairwise':

            os.makedirs(postdisc_path, exist_ok=True)
            nodes_df, clusters_list = run_clustering_pairs(
                matches_df, params['clustering'])


        pickle_save_nodes_clusters(nodes_df, clusters_list, postdisc_path)        
        
    return nodes_df, clusters_list, postdisc_name


def evaluate_discovery(outdir, jobs=1, dataset='phoenix', seq_names=None, cnf=None):
    import subprocess
    from utils.helper_fncs import load_json



    with open(join(outdir, 'seq_names.txt'), 'w') as f: f.write('\n'.join(seq_names))

    # if not os.path.exists(outdir + '/scores.json'):

    cmd = './run_tde.sh {} {} {} {} {} {} {} {}'.format(TDEROOT, outdir, dataset, 
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
   
    matches_df, seq_names = run_matches_discovery(feats_dict, params)
    print('*** found {} matches ***'.format(len(matches_df)))

    nodes_df, clusters_list, postdisc_name = run_clustering(seq_names, matches_df, params)
    print('*** post disc completed, found {} segments from {} clusters ***'.format(len(nodes_df), len(clusters_list)))
    
    scores = evaluate_discovery(join(params['exp_root'],params['expname'], postdisc_name), 
                        jobs=params['njobs'], dataset=params['dataset'],
                        seq_names=seq_names, cnf=params['config_file'])

    # _, olapratio = number_of_discovered_frames(nodes_df, clusters_list, returnolap=True)
    # scores['olapratio'] = olapratio
    scores['length_avg'] = (nodes_df.end - nodes_df.start).mean()
    print('*** Coverage: {:.4f}, NED: {:.2f}'.format(scores['coverageNS'], scores['ned']))
    
    return matches_df, nodes_df, clusters_list, scores



