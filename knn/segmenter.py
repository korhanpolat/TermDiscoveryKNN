import numpy as np
# from numba import jit, njit



# @jit
def pre_seg_indices(a,T,lmin,lmax):
	# generate segmentation intervals 
	# for a sequence of length 'T'
	# in every 'a' points, lengths between 'lmin' - 'lmax'

    n_expected = np.int(((lmax-lmin)*T/(a**2)) + 1) * 5
    intervals = np.zeros((n_expected,2), np.uint64)
    cnt = 0
    for i in range(0,T,a):
        for j in range(i+a,T,a):
            if (j-i >= lmin) and (j-i<=lmax) and (i>=5) and (j<=T-5):  
                intervals[cnt] = (i,j)
                cnt += 1
        # j = T
        # if (j-i >= lmin) and (j-i<=lmax):  
        #     intervals[cnt] = (i,j)
        #     cnt += 1
            
    return intervals[:cnt]


def uniform_segmentation_intervals(feat_dims, a, lmin, lmax):
	# input: length of each seq as dict
	# return: segmentation indices as dict 
	
    intervals_dict = dict()

    for key, T in feat_dims.items():
        intervals_dict[key] = pre_seg_indices(a,T,lmin,lmax)  
        
    return intervals_dict


def uniform_segmenter(feats_dict, params_seg):
    # return indices per seqeuence as dict
    
    feat_dims = {k:v.shape[0] for k,v in feats_dict.items()}
    intervals_dict = uniform_segmentation_intervals(feat_dims, 
                                                    params_seg['a'], 
                                                    params_seg['lminf'], 
                                                    params_seg['lmaxf'])

    return intervals_dict