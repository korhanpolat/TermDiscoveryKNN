from utils.discovery_utils import find_pairwise_overlaps, find_same_match_overlaps, compute_overlap, find_pairwise_overlaps_NMS, find_pairwise_overlaps_NMS_slots
# from numba import jit, njit
import numpy as np
import pandas as pd


# @jit
def sort_as_pairs(D, I, start, end):
    _, k = D.shape

    idx1 = np.repeat( np.arange(start,end,1), k)
    idx2 = I[ start : end ].reshape(-1)
    dists = D[ start : end ].reshape(-1)
    
    args_sorted = dists.argsort()

    return np.vstack([idx1, idx2]).T[args_sorted], dists[args_sorted]


# @jit
def select_tops(D, I, start, end, top_delta):
    
    pairs, dists = sort_as_pairs(D, I, start, end)
    # eliminate self matches
    pairs = np.sort(pairs,axis=1)
    notsame_idx = pairs[:,0] != pairs[:,1]
    idx_select = np.concatenate([~(np.sum((pairs[1:] == pairs[:-1]),1) == 2),[True]] )
    pairs = pairs[idx_select & notsame_idx]
    dists = dists[idx_select & notsame_idx]
    
    n = int(len(pairs)*top_delta)
    pairs, dists = pairs[:n], dists[:n]
    
    return pairs, dists


def retrieve_segment_info(query_id, traceback_info, intervals_dict):
    # segment id to filename and interval
    file_id = np.argmax( traceback_info['idx_cum'] > query_id)

    filekey = traceback_info['fname'][file_id]

    n_previous_embeds = traceback_info['idx'][:file_id].sum()
    interval = intervals_dict[filekey][query_id - n_previous_embeds]
    
    return file_id, filekey, interval


# @jit
def retrieve_pairs_info(pairs, dists, traceback_info, intervals_dict):
    # input:  pairs,
    # returns: arrays of info for non-overlapping pairs
    
    f1f2names = np.zeros(pairs.shape, dtype=object)
    f1f2arr = np.zeros(pairs.shape, dtype=int)
    s1e1s2e2array = np.zeros((len(pairs),4), dtype=int)
#     wgtharray = np.zeros(len(pairs))
    wgtharray = dists

    for p, ids in enumerate(pairs):

#         wgtharray[p] = dists[p]

        for sid, qid in enumerate(ids):
            fileid, filekey, interval = retrieve_segment_info(qid, traceback_info, intervals_dict)
            f1f2names[p,sid] = filekey
            f1f2arr[p,sid] = fileid
            s1e1s2e2array[p,2*sid:2*sid+2] = interval
        
    return f1f2names, f1f2arr, s1e1s2e2array, wgtharray


# @jit
def remove_self_overlaps(info_arrays, olapthr_m):
    to_remove = np.zeros(len(info_arrays), dtype=bool)
    to_remove[find_same_match_overlaps(info_arrays[:,4:6], 
                                       info_arrays[:,6:10], 
                                       info_arrays[:,10], olapthr_m)] = True

    info_arrays = info_arrays[~to_remove]
    
    return info_arrays

# @jit
def remove_pair_overlaps(info_arrays, olapthr_m):
    sort_idx = np.argsort(info_arrays[:,10])
    info_arrays = info_arrays[sort_idx]
    # pairs, f1f2names, f1f2arr, s1e1s2e2array, wgtharray
    to_remove = find_pairwise_overlaps_NMS(np.uint64(info_arrays[:,4:6]), 
                                       np.uint64(info_arrays[:,6:10]), 
                                       np.float64(info_arrays[:,10]).reshape(-1,1), 
                                       np.float64(olapthr_m))

    info_arrays = info_arrays[~to_remove]
    
    return info_arrays


# @jit
def remove_overlaps(info_arrays, olapthr_m):
    
    info_arrays = remove_self_overlaps(info_arrays, 0.1)

    info_arrays = remove_pair_overlaps(info_arrays, olapthr_m)
    
    return info_arrays




def get_non_olap_pairs(pairs, dists, traceback_info, intervals_dict, olapthr_m):
    
    info_arrays = retrieve_pairs_info(
        pairs, dists, traceback_info, intervals_dict)
    
    info_arrays = np.hstack([pairs, np.hstack(info_arrays[:3]), info_arrays[3][:,None] ])
    
    info_arrays = remove_overlaps(info_arrays, olapthr_m)
    
    return info_arrays



def arrays_to_df(arrays_list):
    cols = ['seg_id1','seg_id2',
            'f1','f2',
           'f1_id','f2_id',
           'f1_start', 'f1_end', 'f2_start', 'f2_end',
           'cost']

    return pd.DataFrame(arrays_list, columns=cols)


def pair_selection(D, I, traceback_info, intervals_dict, top_delta, olapthr_m):

    last = 0
    arrays_list = []

    for i,n_seg in enumerate(traceback_info['idx']): # 2.25 sec for 100 iter

        # for each input file, take top delta percent pairs
        pairs, dists = select_tops(D, I, last, last+n_seg, top_delta)
        last += n_seg
        # retrieve back filenames and interval indices, remove overlaps
        info_arrays = get_non_olap_pairs(
                            pairs, dists, traceback_info, intervals_dict, olapthr_m)

        arrays_list.extend(info_arrays)
        
    return arrays_list


def pair_selection_wrap(D, I, traceback_info, intervals_dict, params):

    arrays_list = pair_selection(D, I, traceback_info, intervals_dict, 
                                 params['top_delta'], params['olapthr_m'])
        
    matches_df = arrays_to_df(arrays_list)
    
    return matches_df
