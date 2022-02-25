
import numpy as np
from os.path import join,dirname,abspath
from os import listdir
from sklearn.decomposition import PCA
from scipy.signal import medfilt2d
import os



def apply_PCA(feats, exp_var, whiten=False, unitvar=True):

    X = np.concatenate(feats, axis=0)
    # unit variance each time vector
    if unitvar: X /= X.std(1)[:,None]

    pca = PCA(n_components=exp_var, whiten=whiten, random_state=42)
    pca.fit(X)

    Xt = pca.transform(X)

    res = []
    idx = 0
    for arr in feats:
        n = arr.shape[0]
        res.append(Xt[idx:idx + n, :])
        idx += n

    return res


def apply_PCA_dict(feats_dict, exp_var, whiten=False, unitvar=True):

    names = [] 
    feats = []
    for k,v in feats_dict.items():
        names.append(k)
        feats.append(v)

    res = apply_PCA(feats, exp_var, whiten, unitvar)

    for k,v in feats_dict.items():
        idx = names.index(k)
        feats_dict[k] = res[idx]

    return feats_dict



def op_100_single(name, apply_medfilt=False, feats_root=''):
    ''' load openpose features for single file 
        both hands (dim 2*21) and upper body (dim 8) and x,y coordinates (dim 2) 
        makes (2x21 + 8) * 2 => 100 dimensions
    '''

    arr_pose = np.load(os.path.join(feats_root,'op_body',name + '.npy'))[:,:9,:2]
    shoulder_L = np.linalg.norm(arr_pose[:,5]-arr_pose[:,2],2,1).mean()
    neck = arr_pose[:,1,:2][:,None,:]
    wrist_L = arr_pose[:,7,:2][:,None,:]
    wrist_R = arr_pose[:,4,:2][:,None,:]

    arr_pose = np.delete(arr_pose, 1 , axis=1) # delete the neck coordinate, because uninformative
    
    # normalize body by dividing to shoulder length
    arr = np.hstack(((arr_pose - neck) / shoulder_L,
                     (np.load(os.path.join(feats_root,'op_hand/right',name + '.npy'))[:,:,:2] - wrist_R) / shoulder_L,
                     (np.load(os.path.join(feats_root,'op_hand/left',name + '.npy'))[:,:,:2] - wrist_L) / shoulder_L
                    ))
    arr = arr.reshape(-1,100)
    if apply_medfilt: arr = medfilt2d(arr, kernel_size=(3,1))

    return arr


def op_100(seq_names, as_dict=False, apply_medfilt=False, feats_root=''):
    ''' load openpose features for list of files 
        both hands (dim 2*21) and upper body (dim 8) and x,y coordinates (dim 2) 
        makes (2x21 + 8) * 2 => 100 dimensions
        params:
            seq_names (list of str): filenames to load
            as_dict (bool): whether to load as dict (keys for filenames) or list
            apply_medfilt (bool):   whether to apply median filter temporaly to smooth out  
                                    outliers wrt temporal dimension
    '''
    
    if as_dict: 

        fdict = dict()

        for i,name in enumerate(seq_names):
            fdict[name] = op_100_single(name, apply_medfilt, feats_root)

        return fdict

    else: 
        
        feats = []
        
        for i,name in enumerate(seq_names):
            feats.append( op_100_single(name, apply_medfilt, feats_root) )

        return feats


def get_dh_feats(seq_names, base, hand, params):

    feats_dict = {}
    for name in seq_names: 
        if hand == 'both':
            arrs = [np.load(join(params['feats_root'], '{}/{}/train/'.format(
                                       base,h), name + '.npy')) for h in ['right','left']]
            for a,arr in enumerate(arrs):
                arrs[a] = arr / arr.std(1)[:,None]
            arr = np.concatenate(arrs, axis=1)
        else:
            arr = np.load(join(params['feats_root'], '{}/{}/train/'.format(base,hand), name + '.npy'))

        feats_dict[name] = arr

    return feats_dict


def get_dh_feat_names(featype):
    ''' get path for deephand loader '''
    
    if 'PCA40W' in featype:
        base = 'deep_hand_PCA40W'
    elif 'PCA40' in featype:
        base = 'deep_hand_PCA40'
    else:
        base = 'deep_hand'
    
    if 'c3' in featype:
        base = os.path.join(base , 'c3')
    elif 'l3' in featype:
        base = os.path.join(base , 'l3')

    if 'right' in featype: 
        hand = 'right'
    if 'left' in featype:
        hand = 'left'
    if 'both' in featype:
        hand = 'both'    

    return base, hand
    # return get_dh_feats(seq_names, base, hand, params)

def save_deephand_PCA(seq_names, base, hand, params):

    if 'exp_var' in params.keys(): exp_var = params['exp_var']
    else: exp_var = 0.99

    base_tmp = ''.join(base.split(f'_PCA{exp_var}'))
    feats_dict =  get_dh_feats(seq_names, base_tmp, hand, params)    

    
    feats_dict = apply_PCA_dict(feats_dict, exp_var, 
                        whiten='W' in params['featype'], unitvar='V' in params['featype'])
    
    # save
    os.makedirs(join(params['feats_root'], '{}/{}/train/'.format(base,hand)),exist_ok=True)
    for name, arr in feats_dict.items():
        np.save(join(params['feats_root'], '{}/{}/train/'.format(base,hand), name + '.npy'),
                arr)

    



def load_feats(seq_names, params):
    
    if 'op' in params['featype']:
        feats_dict = op_100(seq_names, as_dict=True, apply_medfilt=True, feats_root=params['feats_root'])
    
    elif ('dh' in params['featype']) or ('c3' in params['featype']):
        base, hand = get_dh_feat_names(params['featype'])

        if ('PCA' in params['featype']): # and (('dh' not in featype) and ('c3' not in featype)):
            fileexists = os.path.isfile( 
                join(params['feats_root'], '{}/{}/train/'.format(base,hand), seq_names[0] + '.npy') )
            if not fileexists:
                print('Could not find saved PCA features, computing now ...')
                save_deephand_PCA(seq_names, base, hand, params)

 
    
        feats_dict =  get_dh_feats(seq_names, base, hand, params)


    return feats_dict
