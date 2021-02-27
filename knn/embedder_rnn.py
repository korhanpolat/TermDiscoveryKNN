# import sys
# sys.path.append('/home/korhan/Desktop/tez/nazif_ae/')
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from torch import nn
from my_models.gcn import EncDecGcnRNN



class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.hidden_size = hidden_size
        self.n_layers = kwargs['rnn_layer_enc']    

#         self.embedding = Encoder(kwargs['input_shape'], hidden_size, **kwargs)
        gru_input_size = kwargs['input_shape']
        self.gru = nn.GRU(gru_input_size, hidden_size, 
                          num_layers=self.n_layers, 
                          dropout=kwargs['dropout'],
                          bidirectional=kwargs['bidirectional'])
        
#         self.final = nn.Sequential( nn.Linear(hidden_size, hidden_size),
#                                     nn.Tanh(),)
        if kwargs['bidirectional']:
            self.n_layers *= 2


    def initHidden(self, batch_size):
        return torch.randn(self.n_layers, batch_size, self.hidden_size, device=self.device)
    
    def forward(self, x, xlen, hidden=None):
        
#         x = self.embedding(x)
        x_packed = pack_padded_sequence(x, xlen, batch_first=True, enforce_sorted=False)
        
        if hidden is None:
            hidden = self.initHidden(x_packed.data.shape[0])

        output, hidden = self.gru(x_packed, hidden)        
#         hidden = self.final(hidden)
                
        return output, hidden[-1]

    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.hidden_size = hidden_size
        self.outsize = kwargs['input_shape']
        self.n_layers = kwargs['rnn_layer_dec']    

        self.gru = nn.GRU( hidden_size, self.outsize, 
                           num_layers=self.n_layers, 
                           dropout=kwargs['dropout'],
                           bidirectional=kwargs['bidirectional'])
#         self.decoder = nn.Sequential(Encoder(hidden_size, self.outsize, **kwargs),
#                                       nn.Linear(self.outsize, self.outsize),
#                                       nn.Tanh()
#                                     )

        if kwargs['bidirectional']:
            self.n_layers *= 2
        
        
    def initHidden(self, batch_size, hidden=None):
        return torch.randn(self.n_layers, batch_size, self.outsize, device=self.device)
    
    def forward(self, h, ylen, hidden=None):
        # x -> h
        
        # repeat hidden vec for seq lengths
        h_repeat = h.repeat(max(ylen),1,1).permute((1,0,2))
        h_packed = pack_padded_sequence(h_repeat, lengths=ylen, batch_first=True, 
                                        enforce_sorted=False)
        
        
        if hidden is None:
            hidden = self.initHidden(h_packed.data.shape[0])

#         print(hidden.shape)
        output, hidden = self.gru(h_packed, hidden)        
      
        output = pad_packed_sequence(output, batch_first=True)

#         output = self.decoder(output)
            
        return output[0], hidden
    
    
# class EncDecRNN(nn.Module):
#     def __init__(self, hidden_size, **kwargs):
#         super().__init__()
#         self.hidden_size = hidden_size
        
#         self.model_encoder = EncoderRNN(int(hidden_size), **kwargs)
#         self.model_decoder = DecoderRNN(hidden_size, **kwargs)       
    
#         # self.pool = nn.MaxPool1d(kwargs['lmaxf'])
#         self.final = nn.Softmax(2)
    
#         self.mode_decode=True
    
#     def forward(self, x, xlen, ylen):
        
#         out, h = self.model_encoder(x, xlen)
# #         print('encode')
        
#         # out_pad = pad_packed_sequence(out, batch_first=True, padding_value=float("-Inf"))
#         # out_pool = self.pool(out_pad[0].permute((0,2,1)))
#         # h = torch.cat([h, out_pool.squeeze()],1)

#         if self.mode_decode:
#             yhat, h_dec = self.model_decoder(h, ylen)
#             # yhat = self.final(yhat)
#             return yhat, h
#         else: 
#             return out, h


class EncDecRNN(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

        
        self.model_encoder = EncoderRNN(int(hidden_size), **kwargs)
        self.model_decoder = DecoderRNN(hidden_size, **kwargs)       
    
        # self.pool = nn.MaxPool1d(kwargs['lmaxf'])
        if kwargs['smax'] :
            self.final = nn.Sequential( nn.Linear(hidden_size, hidden_size),
                                        nn.Softmax(1) )
        else: self.final = None
    
        self.mode_decode=True
    
    def forward(self, x, xlen, ylen):
        
        out, h = self.model_encoder(x, xlen)
#         print('encode')
        if self.final :
            h = self.final(h)
        
        # out_pad = pad_packed_sequence(out, batch_first=True, padding_value=float("-Inf"))
        # out_pool = self.pool(out_pad[0].permute((0,2,1)))
        # h = torch.cat([h, out_pool.squeeze()],1)

        if self.mode_decode:
            yhat, h_dec = self.model_decoder(h, ylen)
            # yhat = self.final(yhat)
            return yhat, h
        else: 
            return out, h


class EmbeddingRNN():
    def __init__(self, params_emb, **kwargs):
        
        self.model_root = '/home/korhan/Desktop/tez/models/'
        self.lmin = kwargs['lminf']
        self.lmax = kwargs['lmaxf']
        self.n = kwargs['dim_fix']
        self.batch = params_emb['batch']
        self.model_name = params_emb['model_name']
        if 'gcn' in self.model_name:
            self.model = EncDecGcnRNN(self.n, **params_emb['model_params']) 
        else:
            self.model = EncDecRNN(self.n, **params_emb['model_params']) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        
        # self.model.load_state_dict(torch.load(kwargs['model_path'] + self.model_name, 
        #                                       map_location=self.device))

        self.model_path = self.model_root + self.model_name


        # self.model = params_emb['model_obj']


        self.model.load_state_dict( torch.load( self.model_path ))
        print('Model: {} loaded, running on {}'.format(self.model_name, self.device))

        # self.model = torch.load(self.model_path) #, map_location=self.device)

        self.model.to(self.device)
        self.model.eval()
        self.model.mode_decode = False

        
    def embed_sequence(self, arr, intervals):
        # given intervals, feature array; output fixed embeddings

        n_db = len(intervals)
        d_fix = self.n
        X_embeds = np.zeros((n_db, d_fix)).astype('float32')
        
        batch = self.batch

        for cnt in range((len(intervals)-1)//batch + 1):
            batch_intervals = intervals[cnt*batch:(cnt+1)*batch]

            xlen = list(batch_intervals[:,1] - batch_intervals[:,0])
            xx = [torch.FloatTensor(arr[i:j]) for i,j in batch_intervals]

            x_pad = pad_sequence(xx, batch_first=True, padding_value=0)
            outputs, code = self.model(x_pad.to(self.device), xlen, xlen)

            X_embeds[cnt*batch:(cnt+1)*batch] = code.squeeze().detach().cpu().numpy()

        # print('Embedding of shape {} computed'.format(X_embeds.shape))

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