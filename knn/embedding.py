import numpy as np
# from numba import jit, njit
from scipy.signal import triang

""" embedding functions """

def g_i(n):
	# variance weigting window for Gaussian interpolation
    return triang(n) * (n/2)


# @jit
def fgaussian(n0, mu, sigma):
	# generate gaussian kernel
    x = np.arange(0,n0)
    kernel = (1/sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
    
    return kernel / np.sum(kernel)


# @jit
def gauss_transformation_mat(n, n0, g, r, s, offs):
	# get transformation matrix
	# params: 
	# 	n 	: target length
	# 	n0	: given length
	# 	r,s : hyper params
	# 	g 	: triangular window	 

    F = np.zeros((n,n0))

    for i in range(n):
        mu = (i*n0)/n
        sigma = r*n0 + s*g[i] + offs
        F[i] = fgaussian(n0, mu, sigma)

    return F
    

def interp_matrices_given_lengths(possible_lengths, n, r, s, offs):
    # return a dict of arrays, each key is length of array to interpolate
    g = g_i(n)
    transmats = dict()

    for n0 in possible_lengths:
        F = gauss_transformation_mat(n, n0, g, r, s, offs)
        transmats[str(n0)] = F

    return transmats


def pre_compute_interp_mats(lmin, lmax, n, r, s, offs):
	# prepare Gaussian smoothing kernels for since possible lengths
	# are determined by the min max interval

#     min_allowd = a*((lmin+a-1)//a)
    possible_lengths = np.arange(lmin,lmax+1,1)
    transmats = interp_matrices_given_lengths(possible_lengths, n, r, s, offs)
        
    return transmats

    

class EmbeddingGaussian():
    def __init__(self, **kwargs):
        
        self.lmin = kwargs['lminf']
        self.lmax = kwargs['lmaxf']
        self.n = kwargs['dim_fix']
        self.r = kwargs['r']
        self.s = kwargs['s']
        if 'offs' in kwargs.keys(): 
            self.offs = self.s * self.n / 2
            self.s = -kwargs['s']

        else: self.offs = 0
        
        self.transmats = pre_compute_interp_mats(self.lmin, self.lmax, 
                                                 self.n, self.r, self.s, self.offs)
        
        
    def embed_segment(self, X):
        
        n0, _ = X.shape

        assert (n0 >= self.lmin ) and (n0 <= self.lmax )
        F = self.transmats[str(n0)]
                
        return np.matmul(F,X).reshape(-1)
        

    def embed_sequence(self, arr, intervals):
        # given intervals, feature array; output fixed embeddings
        
        _, d = arr.shape
        n_db = len(intervals)
        d_fix = self.n * d
        X_embeds = np.zeros((n_db, d_fix)).astype('float32')

        for cnt, (i,j) in enumerate(intervals):
            X_embeds[cnt] = self.embed_segment(arr[i:j])
            
        return X_embeds


    def compute_all_embeddings(self, feats_dict, intervals_dict):
        X_all = []
        traceback_info = dict()
        traceback_info['idx'] =  np.zeros(len(feats_dict), int)
        traceback_info['fname'] = []

        for i, (key, arr) in enumerate(feats_dict.items()):
            X_embeds = self.embed_sequence(arr, intervals_dict[key])
            X_all.append(X_embeds)
            traceback_info['idx'][i] = X_embeds.shape[0]
            traceback_info['fname'].append(key)

        traceback_info['idx_cum'] = np.cumsum(traceback_info['idx'])
        
        return np.concatenate(X_all, axis=0), traceback_info




class EmbeddingSum():
    def __init__(self, **kwargs):

        self.params = kwargs
        

    def embed_segment(self, X):
                   
        return X.sum(0)


    def embed_sequence(self, arr, intervals):
        # given intervals, feature array; output fixed embeddings
        
        _, d = arr.shape
        n_db = len(intervals)
        d_fix = d
        X_embeds = np.zeros((n_db, d_fix)).astype('float32')

        for cnt, (i,j) in enumerate(intervals):
            X_embeds[cnt] = self.embed_segment(arr[i:j])
            
        return X_embeds


    def compute_all_embeddings(self, feats_dict, intervals_dict):
        X_all = []
        traceback_info = dict()
        traceback_info['idx'] =  np.zeros(len(feats_dict), int)
        traceback_info['fname'] = []

        for i, (key, arr) in enumerate(feats_dict.items()):
            X_embeds = self.embed_sequence(arr, intervals_dict[key])
            X_all.append(X_embeds)
            traceback_info['idx'][i] = X_embeds.shape[0]
            traceback_info['fname'].append(key)

        traceback_info['idx_cum'] = np.cumsum(traceback_info['idx'])
        
        return np.concatenate(X_all, axis=0), traceback_info




