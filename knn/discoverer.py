from knn.embedding import EmbeddingGaussian, EmbeddingSum
# from knn.embedder_rnn import EmbeddingRNN
from knn.faissWrapper import KnnIndex, kMeansEmbed
from knn.segmenter import uniform_segmenter
from knn.pair_select import pair_selection_wrap
import os
from copy import deepcopy

def gen_expname(params):
    """Generate name string from experiment parameters. Required for bookkeeping.

    Args:
        params (dict): Experiment parameters

    Returns:
        str: Generated name
    """
    return '_'.join(['{}_{}'.format(key,v) for key,v in params.items()])



class KnnDiscovery():
    """Term discovery using KNN search
    """    
    def __init__(self, feats_dict, params):
        """Initializes Knn discovery object

        Args:
            feats_dict (dict of nd.arrays): Keys are sequence names and values are feature arrays
            params (dict): Experiment parameters
        """        
        self.params = deepcopy(params)
        self.index = None
        self.seq_names = list(feats_dict.keys())

        self.params['disc']['lminf'] = params['disc']['lmin'] * params['disc']['a']
        self.params['disc']['lmaxf'] = params['disc']['lmax'] * params['disc']['a']
        
        if self.params['disc']['seg_type'] == 'uniform':
            # get segmentation intervals per sequence
            self.intervals_dict = uniform_segmenter(feats_dict, 
                                                    self.params['disc'])
            
        if self.params['disc']['emb_type'] in ['gauss_kernel', 'kmeans']:
            self.embedder = EmbeddingGaussian(**self.params['disc'])
            
        # if 'rnn' in self.params['disc']['emb_type']:
        #     self.embedder = EmbeddingRNN(self.params['emb'], **self.params['disc'])

        if self.params['disc']['emb_type'] == 'sum':
            self.embedder = EmbeddingSum(**self.params['disc'])


        self.run_embed(feats_dict) 

        if self.params['disc']['emb_type'] in ['kmeans']:
            kmeans = kMeansEmbed(self.X, **self.params['disc'])
            self.X = kmeans.embed(self.X)
        
            
    def run_embed(self, feats_dict):
        """_summary_

        Args:
            feats_dict (_type_): _description_
        """        
        print('Computing Embeddings')
        self.X, self.traceback_info = self.embedder.compute_all_embeddings(
            feats_dict, self.intervals_dict)
        
        
    def build_index(self, knn_params):
        
        self.index = KnnIndex(self.X, **knn_params)
        
    
    def run_disc(self, new_disc_params=None):

        if type(new_disc_params) is dict: 
            self.params['disc'] = new_disc_params
            self.build_index(self.params['disc'])
        
        if self.index is None: self.build_index(self.params['disc'])
               

        print('Searching index')
        self.D, self.I = self.index.search(self.X, self.params['disc']['k'])
        # self.expname = gen_expname(params['disc'])

        
    def post_disc(self, new_post_params=None):
        if type(new_post_params) is dict: 
            self.params['disc'] = new_post_params
        
        print('Selecting good pairs')
        self.matches_df = pair_selection_wrap(self.D, self.I , 
                                         self.traceback_info, 
                                         self.intervals_dict, 
                                         self.params['disc'])
        
        # self.postdisc_name = gen_expname(params['clustering'])
        
    
    def run(self):
        
        self.run_disc()
        self.post_disc()
        
        return self.matches_df
        