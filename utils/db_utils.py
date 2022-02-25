

import pandas as pd
from os.path import join,dirname,abspath, isfile
import matplotlib.pyplot as plt
from os import listdir
from utils.helper_fncs import load_obj
from matplotlib.image import imread
import numpy as np
import glob


root_dir = '/home/korhan/Dropbox/tez/files/'

"""
class SignerDf(pd.DataFrame):
    @property
    def _constructor(self, signer_id=None, obj_path='/home/korhan/Dropbox/tez'):

        df = pd.read_csv(join(obj_path, 'train_labels.csv'))

        if signer_id is not None:
            df = df.loc[df.signer == signer_id]
            df = df.reset_index()

        return df
"""
#    def get_labels(self):

excluded_labels = [3693, 2370, 2371, 2372, 2385, 2386, 2387]
excluded_labels = [3693]


def get_seq_names(params):
    with open(join(params['CVroot'], params['CVset'] + '.txt'), 'r') as f: 
        seq_names = [x.strip('\n') for x in f.readlines()]

    return seq_names


def imgs_for_foldername(img_root,folder,img_idx):

#    paths = sorted(listdir( join( img_root, folder, '1')) )
#    img = imread(join( img_root, folder, '1', paths[img_idx]))
    paths = sorted(listdir( join( img_root, folder)) )
    img = imread(join( img_root, folder, paths[img_idx]))
    
    return img



def get_labels_dict(path='/media/korhan/ext_hdd/tez_datasets/'):

    if isfile('/home/korhan/Dropbox/tez/files/labels_dict.pkl'):
        return load_obj('labels_dict','/home/korhan/Dropbox/tez/files')

    labels_dict = dict()
    with open(
            path + 'phoenix2014-release/phoenix-2014-multisigner/annotations/automatic/trainingClasses.txt',
            'r') as f:
        cols = f.readline()
        for row in f.readlines():
            row = row.strip('\n').split()
            labels_dict[row[1]] = row[0]

    return labels_dict



def get_labels_for_signer(signer_id):
    """ Signer01    195225
        Signer05    186979
        Signer04    123731
        Signer08     95844
        Signer07     90506
        Signer03     67413
        Signer09     26249
        Signer02      9056
        Signer06      3987 """

    obj_path = '/home/korhan/Dropbox/tez/files'

    df = pd.read_csv(join(obj_path, 'train_labels.csv'))

    if signer_id == 'all':
        return df

    elif type(signer_id) == list:

        df = df.loc[[s in signer_id for s in df.signer]]
        df = df.reset_index()

        return df

    else:
        df = df.loc[df.signer == signer_id]
        df = df.reset_index()

        return df


def get_annots_for_signer(signer_id):

    train_corpus_path = '/home/korhan/Dropbox/tez/files/train.corpus.csv'
    df = pd.read_csv(train_corpus_path, delimiter='|')

    if signer_id == 'all':
        return df

    df = df.loc[df.signer == signer_id]
    df = df.reset_index()

    return df


def annot_for_folder(df, foldername):

    return df.annotation[df.id == foldername]


def img_paths_for_folder(img_root, folder_name, start=0, end=-1):

    img_paths = sorted(glob.glob(join(img_root, folder_name, '*.png')))
    if len(img_paths) == 0:
        img_paths = sorted(glob.glob(join(img_root, folder_name, '1', '*.png' )))

    if end == -1: end=len(img_paths)
#    img_paths = [join(img_root, folder_name, img_path) for img_path in img_paths[start:end]]
    img_paths = img_paths[start:end]

    return img_paths


def imgs_segment(img_root,folder,segment_idx=(0,-1)):

    s,e = segment_idx

    paths = img_paths_for_folder(img_root, folder, start=s, end=e) 
    imgs = []

    for i in range(0,e-s):
        imgs.append( imread(join( img_root, folder, paths[i])) )
    
    return imgs


def info_frames_for_segment(df, foldername, start, end, img_root = '/media/korhan/ext_hdd/tez_datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train'):

    label_names = list(df.label_name[df.folder == foldername][start: end + 1])  # assume frames start from 0
    img_paths = img_paths_for_folder(img_root, foldername + '/1', start, end + 1)


    return label_names, img_paths


def interpolate_garbage_labels(df, excluded_labels=excluded_labels):
    labels_dict = get_labels_dict()

    for filename in df.folder.unique():
        labels = df.label[df.folder == filename]
        labels = ((labels + 1) * labels.apply(lambda x: int(x not in excluded_labels))) - 1
        labels[labels == -1] = None
        df.loc[df.folder == filename, 'label'] = labels.interpolate(method='nearest').ffill().bfill().astype(int)

    df.label_name = df.label.apply(lambda x: labels_dict[str(x)])

    return df


def plot_common_frames_for_pair(df,file_pair):

    """ eg: file_pair(list) = ['28September_2012_Friday_tagesschau_default-18',
             '30September_2012_Sunday_tagesschau_default-16'] """

    import matplotlib.pyplot as plt

    img_root = '/home/korhan/Desktop/tez/dataset/features/fullFrame-210x260px/train'

    common_labels = set(df[df.folder == file_pair[0]].label_name.unique()) & set(
        df[df.folder == file_pair[1]].label_name.unique())
#    garbage_labels = ['__ON__0', '__ON__1', '__ON__2', '__OFF__0', '__OFF__1', '__OFF__2', 'si']

    common_labels = common_labels - set(garbage_labels)

    print(common_labels)

    for seq_name in file_pair:
        print(seq_name)
        labels = (df[df.folder == seq_name].label_name.unique())
        for label in labels:
            if (label in common_labels): print(label)

    common_labels_manual = set()

    for label in common_labels: common_labels_manual = common_labels_manual | {label[:-1]}
    print(common_labels_manual)

    for label in common_labels_manual:
        print(label)
        img_paths = []
        for i, seq_name in enumerate(file_pair):
            frames = df.frame[(df.folder == seq_name) & (df.label_name.isin([label + '0', label + '1', label + '2']))]
            img_paths.append(
                [join(img_root, seq_name, '_'.join(seq_name.split('_')[:-1]) + '.avi_pid0_' + frame) for frame in
                 frames])

        n_cols = max(len(img_paths[0]), len(img_paths[1]))

        figure, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(4 * n_cols, 4), sharey=True)
        figure.set_tight_layout(True)

        for fidx in range(2):
            for j, img_path in enumerate(img_paths[fidx]):
                img = plt.imread(img_path)
                if axes.ndim == 1:
                    ax = axes[fidx]
                else:
                    ax = axes[fidx, j]

                ax.imshow(img)
                ax.set_axis_off()

                ax.set_title(str(label), fontsize=6)


def gold_fragments_df_(signer_id, group=3, interp=False, excluded_labels=excluded_labels):
    # type: (str, int, bool) -> pd.DataFrame

    if interp: grbg ='_no_garbage'
    else: grbg = ''

    # FIXME: excluded labels degistiginde okudugu file degismiyor: FIXED
    filepath = join(root_dir, 'gold_fragmens_' + signer_id + '_group' + str(group) + grbg +'on_off_inc.csv')
    if isfile(filepath):

        gold_df = pd.read_csv(filepath, keep_default_na=False)

        if excluded_labels is not None:
            excluded_labels = [i // group for i in excluded_labels]

            gold_df = gold_df.loc[[(lab not in excluded_labels) for lab in (gold_df.label)]]
            gold_df.reset_index(inplace=True, drop=True)

        return gold_df

    df = get_labels_for_signer(signer_id)
    if interp: df = interpolate_garbage_labels(df)

    fragments_df = pd.DataFrame(columns=['signer_id', 'filename', 'start', 'end', 'label', 'labelname'])
    fragments_df = fragments_df.astype(dtype={'signer_id': str,
                                              'filename': str,
                                              'start': int,
                                              'end': int,
                                              'label': int,
                                              'labelname': str})

    for label_id in range(3693 // group):
        if label_id * group in excluded_labels: continue

        df_sub = df[df.label // group == label_id].reset_index()
        df_sub['frame_id'] = df_sub.frame.apply(lambda x: int(x[4:8]))

        folder_counts = df_sub.folder.value_counts()

        for k in range(len(folder_counts)):
            fragment_df = df_sub[df_sub.folder == folder_counts.keys()[k]]
            if group == 3:
                label_name = fragment_df.label_name.unique()[0][:-1]
            else:
                label_name = fragment_df.label_name.unique()[0]

            splits = np.asarray(np.nonzero(np.asarray(fragment_df.frame_id[:-1]) -
                                           np.asarray(fragment_df.frame_id[1:]) != -1)) + 1
            splits = np.append(splits, len(fragment_df))
#            print('two same label fragments in the same sequence {}'.format(label_id))

            s = 0
            for t in splits:
                e = t
                start = fragment_df.frame_id[s:e].min()
                end = fragment_df.frame_id[s:e].max()
                s = e
                fragments_df = fragments_df.append([{'signer_id' : signer_id,
                                                     'filename': folder_counts.keys()[k],
                                                     'start': start,
                                                     'end': end,
                                                     'label': label_id,
                                                     'labelname': label_name}], ignore_index=True, sort=False)

    fragments_df.to_csv(filepath)

    return fragments_df


def gold_fragments_df_for_signer(signer_id, group=3, interp=False, excluded_labels=excluded_labels):

    if type(signer_id) == list:
        df = pd.DataFrame()
        for s in signer_id:
            df = df.append(gold_fragments_df_(s, group=group, interp=interp, excluded_labels=excluded_labels))

        df.reset_index(inplace=True, drop=True)

        return df

    else:
        return gold_fragments_df_(signer_id, group=3, interp=interp, excluded_labels=excluded_labels)


def fragment_tokenizer(gold_fragments, filename, start, end, boundary_th=0.5):
    " returns grouped labels corresponding to given fragment "

    tmp = gold_fragments[(gold_fragments.filename == filename)].sort_values(by=['start'])
    included_frames = np.maximum(0, np.minimum(tmp.end + 1, end + 1) - np.maximum(tmp.start, start))

    if sum(included_frames > 0) == 1:
        return tmp[included_frames > 0].label.values

    token_lengths = (tmp.end + 1 - tmp.start)

    result =  tmp[((included_frames / token_lengths) > boundary_th)].label.values

#    if len(result) == 0:
#        result = [-1]

    return result


def dominant_label(df, filename, start, end, group, excluded_labels=excluded_labels):
    tmp = df.loc[(df.folder == filename)]
    tmp.reset_index(inplace=True)
    result = (tmp.label[(tmp.index >= start) &
                        (tmp.index <= end) &
                        ([l not in excluded_labels for l in tmp.label])] // group).value_counts().keys()
    if len(result) == 0:
        result = -1
    else:
        result = result[0]

    return result


def nodes_with_info(nodes_df, gold_fragments, interp=False):

    nodes_df['signer_id'] = nodes_df.filename.apply(lambda x: gold_fragments.signer_id[gold_fragments.filename == x].unique()[0])
    if interp: nodes_df['label_name'] = nodes_df['labels_dom'].apply(
        lambda x: gold_fragments['labelname'][gold_fragments['label'] == x].unique()[0])

    return  nodes_df


def nodes_with_types(nodes_df, gold_fragments, thr=0.5):

    types_list = []
    for i, row in nodes_df.iterrows():
        types_list.append(tuple(fragment_tokenizer(gold_fragments, row.filename, row.start, row.end, boundary_th=thr)))
    nodes_df['types'] = types_list

    return nodes_df

# FIXME: excluded labels icin tek bir yeri degistir global var olsun

def nodes_with_dom_labels(df, nodes_df, group=3, excluded_labels=excluded_labels):

    types_list = []
    for i, row in nodes_df.iterrows():
        types_list.append( dominant_label(df, row.filename, row.start, row.end, group, excluded_labels))
    nodes_df['labels_dom'] = types_list

    return nodes_df


def gold_matches(df, gold_df, seq_names):

    nodes_df = pd.DataFrame(columns=gold_df.columns)
    clusters = []
    cnt = 0
    matching_labels = set(gold_df.labelname)
    for seq in seq_names:
        matching_labels &= set(gold_df.labelname[ (gold_df.filename == seq )] )
    for label in matching_labels:
        clus = []
        for seq in seq_names:
            nodes_df = nodes_df.append(gold_df.loc[ (gold_df.filename == seq) & (gold_df.labelname == label) ] ,
                                       ignore_index=True)
            clus.append(cnt)
            cnt += 1
        clusters.append(clus)
    nodes_df.reset_index(inplace=True, drop=True)

    return nodes_df, clusters