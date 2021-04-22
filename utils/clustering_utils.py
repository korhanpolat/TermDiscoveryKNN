from os.path import join
import numpy as np
import os
import pandas as pd
from utils.helper_fncs import save_obj, load_obj
import pickle
from numba import jit, njit
import numba as nb
from utils.discovery_utils import compute_overlap


def apply_cost_edge_w(matches_df, cost_thr=0.1):
    # apply threshold to normalized edge weights
    cost_max = matches_df.cost.max()
    matches_df['edge_w'] = matches_df['cost'].apply(lambda x: (cost_max - x) / cost_max)
    
    cost_val = matches_df['edge_w'].sort_values(ascending=False).values[round(len(matches_df) * cost_thr)]
    matches_df = matches_df.loc[matches_df['edge_w'] > cost_val].copy()
    matches_df.reset_index(inplace=True, drop=True)

    return matches_df


def prune_clusters(clusters_list_tmp, cluster_thr=2):
    # need to remove singletons after clustering is performed, can also be thresholded

    clusters_list = []
    excluded_clusters = []
    excluded_nodes = []
    all_nodes = set()

    for i, cluster in enumerate(clusters_list_tmp):

        all_nodes |= set(cluster)

        if len(cluster) < cluster_thr:
            excluded_clusters.append(i)
            excluded_nodes.extend(cluster)
        else:
            clusters_list.append(cluster)

    included_nodes = sorted( list(all_nodes - set(excluded_nodes)) )

    return clusters_list, excluded_clusters, included_nodes, excluded_nodes



def pickle_load_nodes_clusters(postdisc_path):
    nodes_df = pd.read_pickle(join(postdisc_path,'nodes.pkl'))
    clusters_list = load_obj(name='clusters', path=postdisc_path)
    return nodes_df, clusters_list


def pickle_save_nodes_clusters(nodes_df, clusters_list, postdisc_path):
    nodes_df.to_pickle(join(postdisc_path,'nodes.pkl'), protocol=3)
    save_obj(name='clusters', path=postdisc_path, obj=clusters_list)



def dedup_clusters(clusters_list, nodes_df, dedupthr=0.5, minclussize=2):

    for c,cluster in enumerate(clusters_list):
        n = 0
        while n < len(cluster):

            to_remove = []
            i1 = int(cluster[n])

            for m in range(n+1,len(cluster)):
                i2 = int(cluster[m])

                if nodes_df.filename[i1] == nodes_df.filename[i2]:

                    xA = nodes_df.start[i1]
                    xB = nodes_df.end[i1]
                    yA = nodes_df.start[i2]
                    yB = nodes_df.end[i2]

                    folap = compute_overlap(xA, xB, yA, yB)

                    if folap >= dedupthr:
                        to_remove.append(i2)
            n += 1

            to_remove.reverse()
            for i2 in to_remove:
                cluster.remove(i2)

            if len(cluster) >= minclussize:
                clusters_list[c] = cluster
            else:
                del clusters_list[c]
                
    return clusters_list


def matches_to_nodes(matches_df):
    # convert to nodes df
    tmp1 = matches_df[['f1','f1_start','f1_end']]
    tmp1.reset_index(inplace=True, drop=True)
    tmp1.index = tmp1.index * 2
    tmp1.columns = ['filename','start','end']

    tmp2 = matches_df[['f2','f2_start','f2_end']]
    tmp2.reset_index(inplace=True, drop=True)
    tmp2.index = tmp2.index * 2 + 1
    tmp2.columns = ['filename','start','end']

    nodes_df = pd.concat([tmp1,tmp2]).sort_index()
    
    return nodes_df

@njit
def add_pair_edges(seqnameidx, s1e1s2e2, weights):
    # add pairwise edges
    
    n = len(s1e1s2e2)
    edges_mat = np.zeros((2*n, 2*n), dtype=np.float32)
    
    for i,  edge_w in enumerate(weights):

        f1,f2 = seqnameidx[i]
        s1,e1,s2,e2 = s1e1s2e2[i]

        idx1, idx2 = 2*i, 2*i+1

        edges_mat[idx1,idx2] += edge_w

    return edges_mat

@njit
def add_olap_edges(seqnameidx, s1e1s2e2):
    # add overlap edges
    
    n = len(s1e1s2e2)
    edges_mat = np.zeros((2*n, 2*n), dtype=np.float32)

    for i in range(n):

        s1i,e1i,s2i,e2i = s1e1s2e2[i]
        f1i,f2i = seqnameidx[i]

        for j in range(i+1,n):

            s1j,e1j,s2j,e2j = s1e1s2e2[j]
            f1j,f2j = seqnameidx[j]

            edges_mat[2*i,2*j] += compute_overlap(s1i, e1i, s1j,e1j) * (f1i == f1j)
            edges_mat[2*i+1,2*j] += compute_overlap(s2i, e2i, s1j,e1j) * (f2i == f1j)
            edges_mat[2*i,2*j+1] += compute_overlap(s1i, e1i, s2j,e2j) * (f1i == f2j)
            edges_mat[2*i+1,2*j+1] += compute_overlap(s2i, e2i, s2j,e2j) * (f2i == f2j)

        return edges_mat
    
def matches_to_edges_mat(matches_df, mix_ratio):

    seqnameidx = matches_df[['f1_id','f2_id']].values
    s1e1s2e2 = matches_df[['f1_start','f1_end','f2_start','f2_end']].values
    weights = (matches_df['cost'].max() - matches_df['cost'].values) / matches_df['cost'].max()

    edges_olap = add_olap_edges(seqnameidx, s1e1s2e2)

    edges_pair = add_pair_edges(seqnameidx, s1e1s2e2, weights)

    edges_mat = mix_ratio * edges_pair + (1-mix_ratio) * edges_olap
    
    return edges_mat