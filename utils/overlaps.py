# import glob
from os.path import join
import numpy as np
import os
# import pandas as pd
# from utils.helper_fncs import save_obj, load_obj
# import pickle
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


