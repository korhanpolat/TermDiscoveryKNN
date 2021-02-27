import glob
from os.path import join
import numpy as np
import os
import pandas as pd
from utils.helper_fncs import save_obj, load_obj
import pickle
from numba import jit, njit
import numba as nb

@njit(nb.f8(nb.u8, nb.u8, nb.u8, nb.u8))
def compute_overlap(xA, xB, yA, yB):
    # x,y files, A start, B end
    num = yB-yA + xB-xA
    den = max(xB,yB) - min(xA,yA)
    folap = max(0,num/den-1)
    
    return folap


@njit(nb.f8(nb.u8, nb.u8, nb.u8, nb.u8))
def compute_intersect(xA, xB, yA, yB):
    num = yB-yA + xB-xA
    den = max(xB,yB) - min(xA,yA)
    intersct = max(num - den, 0)
    
    return intersct


@njit(nb.b1[:](nb.u8[:, :], nb.u8[:, :], nb.f8[:, :], nb.f8))
def find_pairwise_overlaps(f1f2arr, s1e1s2e2array, wgtharray, olapthr):
    n = len(f1f2arr)
    to_remove = np.zeros(n, dtype=np.bool8)
       
    for i in range(n):
        if1, if2 = f1f2arr[i]
        i1s, i1e, i2s, i2e = s1e1s2e2array[i]
        icost = wgtharray[i][0]

        for j in range(i+1,n):
            if to_remove[j]: continue

            jf1, jf2 = f1f2arr[j]
            j1s, j1e, j2s, j2e = s1e1s2e2array[j]
            jcost = wgtharray[j][0]

            if (if1 == jf1) & ((if2 == jf2)): # i1-j1, i2-j2
                folap1 = compute_overlap(i1s,i1e, j1s, j1e)
                folap2 = compute_overlap(i2s,i2e, j2s, j2e)    
            elif (if1 == jf2) & ((if2 == jf1)): # i1-j2, i2-j1
                folap1 = compute_overlap(i1s,i1e, j2s, j2e)
                folap2 = compute_overlap(i2s,i2e, j1s, j1e)
            else:
                continue

            if (folap1 > olapthr) & (folap2 > olapthr):
                
                if icost > jcost: rmv_idx = i
                else: rmv_idx = j
                
                to_remove[rmv_idx] = True

    return to_remove



@njit(nb.b1[:](nb.u8[:, :], nb.u8[:, :], nb.f8[:, :], nb.f8))
def find_pairwise_overlaps_NMS(f1f2arr, s1e1s2e2array, wgtharray, olapthr):
    n = len(f1f2arr)
    to_remove = np.zeros(n, dtype=np.bool8)

    for i in range(n):
        if to_remove[i]: continue
        if1, if2 = f1f2arr[i]
        i1s, i1e, i2s, i2e = s1e1s2e2array[i]
        icost = wgtharray[i][0]

        for j in range(i+1,n):
            if to_remove[j]: continue

            jf1, jf2 = f1f2arr[j]
            j1s, j1e, j2s, j2e = s1e1s2e2array[j]
            jcost = wgtharray[j][0]

            if (if1 == jf1) & ((if2 == jf2)): # i1-j1, i2-j2
                folap1 = compute_intersect(i1s,i1e, j1s, j1e)
                folap2 = compute_intersect(i2s,i2e, j2s, j2e)    
                
            elif (if1 == jf2) & ((if2 == jf1)): # i1-j2, i2-j1
                folap1 = compute_intersect(i1s,i1e, j2s, j2e)
                folap2 = compute_intersect(i2s,i2e, j1s, j1e)
            else:
                continue
                
            denom = (i1e - i1s) * (i2e - i2s)
            folap = (folap1*folap2) / denom

            if (folap > olapthr):
                rmv_idx = j
                to_remove[rmv_idx] = True

    return to_remove


@njit(nb.b1[:](nb.u8[:, :], nb.u8[:, :], nb.f8[:, :], nb.f8))
def find_pairwise_overlaps_NMS_slots(f1f2arr, s1e1s2e2array, wgtharray, olapthr):

    nsentence = int(f1f2arr.max() + 1)
    
    n = len(f1f2arr)
    to_remove = np.zeros(n, dtype=np.bool8)


    for si in range(nsentence ):
        for sj in range(si, nsentence):
            indices_bool = np.logical_or( np.logical_and( f1f2arr[:,0] == si, f1f2arr[:,1] == sj) ,
                                     np.logical_and( f1f2arr[:,1] == si, f1f2arr[:,0] == sj)  )

            indices = np.nonzero(indices_bool)[0]


            for ik in range(len(indices)):
                i = indices[ik]
                if to_remove[i]: continue
                
                if1, if2 = f1f2arr[i]
                i1s, i1e, i2s, i2e = s1e1s2e2array[i]
                icost = wgtharray[i][0]

                for jk in range(i+1, len(indices)):
                    j = indices[jk]
                    if to_remove[j]: continue

                    jf1, jf2 = f1f2arr[j]
                    j1s, j1e, j2s, j2e = s1e1s2e2array[j]
                    jcost = wgtharray[j][0]

                    if (if1 == jf1) & ((if2 == jf2)): # i1-j1, i2-j2
                        folap1 = compute_intersect(i1s,i1e, j1s, j1e)
                        folap2 = compute_intersect(i2s,i2e, j2s, j2e)    

                    elif (if1 == jf2) & ((if2 == jf1)): # i1-j2, i2-j1
                        folap1 = compute_intersect(i1s,i1e, j2s, j2e)
                        folap2 = compute_intersect(i2s,i2e, j1s, j1e)
                    else:
                        continue

                    denom = (i1e - i1s) * (i2e - i2s)
                    folap = (folap1*folap2) / denom

                    if (folap > olapthr):
                        rmv_idx = j
                        to_remove[rmv_idx] = True


    return to_remove


# @jit
def find_same_match_overlaps(f1f2arr, s1e1s2e2array, wgtharray, olapthr=0.2):
    to_rmv_idx = []
    # samefile = f1f2arr[:,0] == f1f2arr[:,1]
    for i in range(len(f1f2arr)):
        if f1f2arr[i,0] == f1f2arr[i,1]:
            xA, xB, yA, yB = s1e1s2e2array[i]
            olap = compute_overlap(xA, xB, yA, yB)
            if olap > olapthr: to_rmv_idx.append(i)
                
    return to_rmv_idx



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