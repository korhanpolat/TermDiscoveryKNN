import faiss
from faiss import index_factory

class KnnIndex():
    def __init__(self, X, **kwargs):

        
        self.metric_dict = {'L2': faiss.METRIC_L2,
                            'IP': faiss.METRIC_INNER_PRODUCT}
        
        self.metric = kwargs['metric']  # ['L2','IP']
        self.pca = kwargs['pca']        # ['', 'PCA{d}', 'PCAW{d}']
        self.gpu = kwargs['use_gpu']    # [True, False]
        self.normalize = kwargs['norm']
        
        self.build_train(X)

        
    def build_train(self, X):
        
        X = X.astype('float32')
        
        if len(X.shape) > 2: X = X.reshape(X.shape[0],-1)        
        N, d_fix = X.shape
        
        if self.normalize: faiss.normalize_L2(X)

        print('Building index of size {}x{}'.format(N, d_fix))
        
        # build a CPU index
        self.index = index_factory(d_fix, 
                                   "{},Flat".format(self.pca), 
                                   self.metric_dict[self.metric])

        if self.gpu:
            res = faiss.StandardGpuResources()  # use a single GPU

            res.setTempMemory(N * d_fix * 4 *2) # allocate GB 
            # make it into a gpu index
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # train index
        if not self.index.is_trained: self.index.train(X)

        
    def search(self, X, k, Xq=None):

        k = int(k)

        X = X.astype('float32')
        if self.normalize: faiss.normalize_L2(X)
    
        self.index.add(X)
        
        assert self.index.is_trained
        
        if Xq is None: Xq = X
        # search Xq in index
        D, I = self.index.search(Xq, k)
        
        return D, I


class kMeansEmbed():
    def __init__(self, x, **kwargs):
        self.ncentroids = kwargs['ncentroids']
        niter = 20
        verbose = True
        d = x.shape[1]
        self.kmeans = faiss.Kmeans(d, self.ncentroids, niter=niter, verbose=verbose, gpu=True)
        self.kmeans.train(x)        
        
    def embed(self, x):
        
        D, I = self.kmeans.index.search(x, self.ncentroids)
        
        for i, dist_vec in enumerate(D):
            D[i] = dist_vec[I[i]]
            
        return D