"""Contains methods for pairwise post-processing.

This is actually not a clustering in the sense that clusters contain only two segments. 
Each pair is treated as a cluster on its own. 
However, it can be used instead of community detection clustering, hence the same notation 
for nodes and clusters are followed for the sake of completeness.

Typical usage example:

    nodes_df, clusters_list = run_clustering_pairs(matches_df, params['clus'])

"""

import numpy as np
import pandas as pd
# import logging

from clustering.shared_methods import (apply_cost_edge_w, nodes_starts_from1)
from utils.overlaps import find_pairwise_overlaps_NMS, find_same_match_overlaps


def matches_to_arrays(matches_thd, cols):
    """ Convert df to arrays for faster search

    Args:
        matches_thd (pandas.DataFrame): thresholded matches dataframe of length N
        cols (list of str): names of similarity weight columns

    Returns:
        tuple containing

        - fnames (list of str): sorted list of folder names
        - f1f2arr (np.array[Nx2]): indices of filenames for first and second segment w.r.t. sorted 'fnames'
        - s1e1s2e2array (np.array[Nx4]): start and end indices of first and second segment
        - wgtharray (np.array[Nx1]): similarity weights between pairs of segments
    """    
    fnames = sorted(list(set(matches_thd.f1.unique()) | set(matches_thd.f2.unique())))

    f1f2arr = np.column_stack([matches_thd[fs].apply(lambda x: fnames.index(x)) for fs in ['f1','f2']])
    s1e1s2e2array = matches_thd[['f1_start','f1_end' ,'f2_start','f2_end']].values
    wgtharray = matches_thd[cols].values

    return fnames, np.uint64(f1f2arr), np.uint64(s1e1s2e2array), np.float64(wgtharray)


def deduplicate_matches(matches_df, params_clus):
    """Remove overlaps using Non-Max Suppression.

    Args:
        matches_df (pandas.DataFrame): Matches dataframe
        params_clus (dict): Clustering parameters

    Returns:
        pandas.DataFrame: Matches dataframe without overlapping segments
    """    

    # convert to arrays for speed
    fnames, f1f2arr, s1e1s2e2array, wgtharray = matches_to_arrays(matches_df, cols=['cost'])
    # find indices to remove:
    # for overlaps across pairs
    to_remove = find_pairwise_overlaps_NMS(f1f2arr, s1e1s2e2array, wgtharray, params_clus['olapthr_m'])
    # for same pair overlaps 
    to_remove[find_same_match_overlaps(f1f2arr, s1e1s2e2array, wgtharray, params_clus['olapthr_m'])] = True
    # clean the df
    matches_df.drop(np.nonzero(to_remove)[0], inplace=True)
    
    return matches_df.reset_index()


def match_pairs_as_clusters(matches_df):
    """Convert matches to nodes and clusters. 

    Args:
        matches_df (pandas.DataFrame): _description_

    Returns:
        tuple containing

        - nodes_df (pandas.DataFrame): Contains info of all segments
        - clusters_list (list of lists of int): Each sublist is a cluster and 
            contains indices of nodes for that cluster
    """    
    nodes_list = []
    clusters_list = []

    for i,row in matches_df.iterrows():
        nodes_list.append((row['f1'], row['f1_start'], row['f1_end']))
        nodes_list.append((row['f2'], row['f2_start'], row['f2_end']))

        clusters_list.append([2*i,2*i+1])

    nodes_df = pd.DataFrame(nodes_list, columns=['filename','start','end'] )
    
    return nodes_df, clusters_list


def run_clustering_pairs(matches_df, params_clus):
    """Runs post-processing step such that each cluster contains pairs.

    Args:
        matches_df (pandas.DataFrame): Discovered pairwise matches
        params_clus (dict): Clustering parameters

    Returns:
        tuple containing

        - nodes_df (pandas.DataFrame): Contains info of all segments
        - clusters_list (list of lists of int): Each sublist is a cluster and 
            contains indices of nodes for that cluster
    """
    print('*** pairwise clustering ***')

    # apply cost threshold to eliminate low similarity matches
    matches_df = apply_cost_edge_w(matches_df, params_clus['cost_thr'])
    matches_df = matches_df.sort_values(by='cost').reset_index(drop=True)
    # remove overlaps
    matches_df = deduplicate_matches(matches_df, params_clus)

    nodes_df, clusters_list = match_pairs_as_clusters(matches_df)

    nodes_df, clusters_list = nodes_starts_from1(nodes_df, clusters_list)

    return nodes_df, clusters_list