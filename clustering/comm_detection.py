import numpy as np
import igraph as ig
from numba import jit, prange
import sys
sys.path.append('../')
from shared_methods import apply_cost_edge_w, prune_clusters, dedup_clusters, nodes_starts_from1
from itertools import combinations
import pandas as pd


def make_symmetric(edges_mat):

    edges_mat += edges_mat.T
    # return edges_mat
    return np.triu(edges_mat,1)


def edges2_vertices(edges_mat):

    edges_nz = edges_mat.nonzero()
    
    weights = edges_mat[edges_nz]
    edges = [(edges_nz[0][t], edges_nz[1][t]) for t in range(len(edges_nz[0]))]
        
    # iGraph expects vertex names to be 0,1,2,..., so we need a mapping
    vertices = set()
    for e in edges: vertices |= set(e)
    vertex_mapping = list(vertices)
#    vertex_mapping.index(7)
    # update edges according to new mapping
    edges = [(vertex_mapping.index(e[0]), vertex_mapping.index(e[1])) for e in edges]

    return edges, weights, vertex_mapping


def edges_to_iGraph(edges_mat):

    edges, weights, vertex_mapping = edges2_vertices(edges_mat)

    g = ig.Graph()

    g.add_vertices(vertex_mapping)  # add a list of unique vertices to the graph
    g.add_edges(edges)  # add the edges to the graph..
    g.es['weight'] = weights

    g = g.simplify(combine_edges=sum)

    return g, vertex_mapping


def memberships_to_cluster_list(memberships):
    memberships = np.asarray(memberships)

    clusters_list = []
    for c in np.unique(memberships):
        clusters_list.append(np.nonzero(memberships == c)[0].tolist())
    return clusters_list


def cluster_adj_mat(edges_mat, q_thr=0.8, method='fastgreedy'):
    # weights = edges_mat[edges_mat.nonzero()]  # edge weights

    g, vertex_mapping = edges_to_iGraph(edges_mat)
    print('*** vertex mapping done ***')

    if method=='fastgreedy':
        dend = g.community_fastgreedy()


        if q_thr == 0:
            optimal_count = dend.optimal_count
        else:
            # func. below is added from: ...envs/tez/lib/python3.6/site-packages/igraph/clustering.py
            optimal_count = dend.optimal_count_ratio(ratio=q_thr)

        clus = dend.as_clustering(optimal_count)

    elif method=='louvain':
        clus = g.community_multilevel()

    memberships = clus.membership

    modularity = clus.modularity

    clusters_list = memberships_to_cluster_list(memberships)

    # 
    for i,clus in enumerate(clusters_list):
        clusters_list[i] = [ vertex_mapping[c]  for c in clus]

    return clusters_list, memberships, modularity


def find_single_nodes_in_matrix(edges_mat):
    # find isolated nodes in the graph
    assert edges_mat.shape[0] == edges_mat.shape[1] # i.e. square
    nodes_to_delete = []
    for i in range(edges_mat.shape[0]):
        if sum(edges_mat[i,:]) == 0 and sum(edges_mat[:,i]) == 0:
            nodes_to_delete.append(i)
    return nodes_to_delete


def remove_single_nodes(edges_mat, nodes_to_delete):
    # delete nodes that dont have any connection
    edges_mat = np.delete(edges_mat, np.asarray(nodes_to_delete), axis=0 )
    edges_mat = np.delete(edges_mat, np.asarray(nodes_to_delete), axis=1 )

    return edges_mat


# def deduplicate_matches(matches_df, params_clus):
#     # apply cost threshold
#     matches_df = apply_cost_edge_w(matches_df, params_clus['cost_thr'])
#     matches_df = matches_df.sort_values(by='cost').reset_index(drop=True)
#     # convert to arrays for speed
#     fnames, f1f2arr, s1e1s2e2array, wgtharray = matches_to_arrays(matches_df, cols=['cost'])
#     # find indices to remove
#     to_remove = find_pairwise_overlaps_NMS(f1f2arr, s1e1s2e2array, wgtharray, params_clus['olapthr_m'])
#     # find same file overlaps too
#     to_remove[find_same_match_overlaps(f1f2arr, s1e1s2e2array, wgtharray, params_clus['olapthr_m'])] = True
#     # clean the df
#     matches_df.drop(np.nonzero(to_remove)[0], inplace=True)
    
#     return matches_df.reset_index()


def compute_similarity_profile(matches_df, seq_names):
    # find the similarity profile, updated

    max_frame = matches_df[['f1_end','f2_end']].max().max() + 1
    similarity_profile = dict()

    for i, seq_name in enumerate(sorted(seq_names)):
        similarity_profile[seq_name] = np.zeros(max_frame)

    for i, seq_name in enumerate(sorted(seq_names)):

        for j in range(1,3):

            s = 'f' + str(j)+'_start'
            e = 'f' + str(j)+'_end'

            tmp = matches_df[[s,e]][ matches_df['f' + str(j) ] == seq_name ]
            ew = 'edge_w'

            tmp = matches_df[[s, e, ew]][ matches_df['f' + str(j) ] == seq_name ]

            for ri, row in tmp.iterrows():
                similarity_profile[seq_name][int(row[s]):int(row[e])] += row[ew]

    return similarity_profile


from scipy.signal import triang, convolve, find_peaks
def find_node_centers(similarity_profile, peak_thr=1, filter_w=13):

    node_centers = dict()

    triang_w = triang(filter_w) / (filter_w/2)

    for name, vec in similarity_profile.items():

        filtered_vec = convolve(vec, triang_w, mode='same' )
        peaks = find_peaks(filtered_vec, height=peak_thr)
        node_centers[name] = peaks[0]

    return node_centers



def nodes_list_to_df(node_centers):
    # expand note centers to a list
    nodes_centers_list = [] #pd.DataFrame(columns=['name','idx'])
    for name, vec in node_centers.items():
        for i in vec:
            nodes_centers_list.append({'filename':name, 'idx':i})

    nodes_centers_df = pd.DataFrame.from_records(nodes_centers_list, columns=['filename','idx'])

    return nodes_centers_df



@jit
def find_included_node_centers_fast(node_centers_array,f1,f2, s1,e1,s2,e2):
      
    nidx1 = np.where([(node_centers_array[:,0] == f1) & (node_centers_array[:,1]>s1)& (node_centers_array[:,1]<e1) ])[1]
    nidx2 = np.where([(node_centers_array[:,0] == f2) & (node_centers_array[:,1]>s2)& (node_centers_array[:,1]<e2) ])[1]
    
    return nidx1, nidx2


@jit
def compute_adjacency_matrix_fast_engine(n, node_centers_array, seqnameidx, s1e1s2e2, weights ):
    
    edges_mat = np.zeros((n, n))
    path_idx_cnt = np.zeros((n, n), dtype=np.int32)
    path_idx_mat = np.zeros((n, n, 40), dtype=np.int32)  # [idx1,idx2,cnt] -> row i


    for i,  edge_w in enumerate(weights):
        
        f1,f2 = seqnameidx[i]
        s1,e1,s2,e2 = s1e1s2e2[i]
        
        idx1, idx2 = find_included_node_centers_fast(node_centers_array, f1,f2, s1,e1,s2,e2)

        if len(idx1) == 0 or len(idx2) == 0: continue

        edges_mat[np.ix_(idx1,idx2)] += edge_w

        path_idx_mat[np.ix_(idx1,idx2, path_idx_cnt[np.ix_(idx1,idx2)].reshape(-1))] = i
        path_idx_mat[np.ix_(idx2,idx1, path_idx_cnt[np.ix_(idx2,idx1)].reshape(-1))] = i

        path_idx_cnt[np.ix_(idx1,idx2)] += 1
        path_idx_cnt[np.ix_(idx2,idx1)] += 1

        
    return edges_mat, path_idx_mat, path_idx_cnt



def compute_adjacency_matrix_fast(seq_names, nodes_centers_df, matches_df):

    n = len(nodes_centers_df)
    
    seq_names = sorted(seq_names)

    # convert node centers to array
    filesidx = nodes_centers_df.filename.apply(lambda x: seq_names.index(x)).values
    centers = nodes_centers_df.idx.values
    node_centers_array = np.vstack([filesidx,centers]).T # shape nnode x 2

    # convert matches to arrays too
    seq1nameidx = matches_df.f1.apply(lambda x: seq_names.index(x)).values
    seq2nameidx = matches_df.f2.apply(lambda x: seq_names.index(x)).values
    seqnameidx = np.vstack([seq1nameidx, seq2nameidx]).T

    s1e1s2e2 = matches_df[['f1_start','f1_end','f2_start','f2_end']].values
    weights = matches_df['edge_w'].values

    edges_mat, path_idx_mat, path_idx_cnt = compute_adjacency_matrix_fast_engine(
        n, node_centers_array, seqnameidx, s1e1s2e2, weights )

    return edges_mat, path_idx_mat, path_idx_cnt



def remove_isolated_nodes(edges_mat, path_idx_mat, path_idx_cnt, nodes_centers_df):

    nodes_to_delete = find_single_nodes_in_matrix(edges_mat)

    if len(nodes_to_delete) > 0:
#        print('{} nodes are deleted..'.format(len(nodes_to_delete)))
        edges_mat = remove_single_nodes(edges_mat, nodes_to_delete)
        path_idx_mat = remove_single_nodes(path_idx_mat, nodes_to_delete)
        path_idx_cnt = remove_single_nodes(path_idx_cnt, nodes_to_delete)

        # also need to delete those nodes from node centers df
        nodes_centers_df.drop(nodes_to_delete, inplace=True)
        nodes_centers_df.reset_index(inplace=True, drop=True)

    return edges_mat, path_idx_mat, path_idx_cnt, nodes_centers_df



def determine_node_intervals(matches_df, nodes_centers_df, clusters_list, path_idx_mat, path_idx_cnt, included_nodes):
    # find intervals of nodes that share a common path and are in the same cluster

    intervals = np.zeros((len(nodes_centers_df), 3))

    for clus in clusters_list:

        for pair in combinations(clus, 2):  # same cluster

            n_center_for_pair = path_idx_cnt[pair]

            #        if n_center_for_pair > 0:  # if path exists between the two nodes

            for k in range(n_center_for_pair):

                row = matches_df.loc[path_idx_mat[pair][k]]

                if row['f1'] == nodes_centers_df.filename[pair[0]] and row['f2'] == nodes_centers_df.filename[pair[1]]:
                    ix1, ix2 = pair
                elif row['f2'] == nodes_centers_df.filename[pair[0]] and row['f1'] == nodes_centers_df.filename[pair[1]]:
                    ix2, ix1 = pair
                # else:
                    # pdb.set_trace()

                    # raise Exception('Filenames of pair does not match!!')

                intervals[ix1, 0] += row['f1_start']
                intervals[ix1, 1] += row['f1_end']
                intervals[ix1, 2] += 1
                intervals[ix2, 0] += row['f2_start']
                intervals[ix2, 1] += row['f2_end']
                intervals[ix2, 2] += 1

    intervals = intervals[included_nodes]
    start_indices = np.int64(intervals[:, 0] / intervals[:, 2])
    end_indices = np.int64(intervals[:, 1] / intervals[:, 2])

    return start_indices, end_indices


def get_segment_info(matches_df, center):
    tmp_df = matches_df.loc[(matches_df['f1'] == center[0].filename) & 
               (matches_df['f1_start'] <= center[0].idx) & 
               (matches_df['f1_end'] >= center[0].idx) &
               (matches_df['f2'] == center[1].filename) & 
               (matches_df['f2_start'] <= center[1].idx) & 
               (matches_df['f2_end'] >= center[1].idx)]

    return tmp_df


def update_interval(intervals, ix, val ):
    s, e, w = val   
    
    intervals[ix,0] += s * w
    intervals[ix,1] += e * w
    intervals[ix,2] += w
    

def update_interval_pair(intervals, pair, row):
    
    ix1, ix2 = pair

    update_interval(intervals, ix1, row[['f1_start','f1_end','edge_w']].values)
    update_interval(intervals, ix2, row[['f2_start','f2_end','edge_w']].values)
    

def determine_node_intervals_new(matches_df, nodes_centers_df, clusters_list, included_nodes):
    
    intervals = np.zeros((len(nodes_centers_df), 3) )

    for clus in clusters_list:

        for pair in combinations(clus, 2):  # same cluster
            pair = list(pair)

            center = [nodes_centers_df.loc[p] for p in pair]
            segs = get_segment_info(matches_df, center)

            if len(segs) == 0: 
                center.reverse()
                pair.reverse()
                segs = get_segment_info(matches_df, center)

            for s, row in segs.iterrows():
                update_interval_pair(intervals, pair, row)

    assert list(np.where(np.any(intervals, axis=1))[0]) == included_nodes 

    intervals = intervals[included_nodes]
    start_indices = np.uint64(intervals[:, 0] / intervals[:, 2])
    end_indices = np.uint64(intervals[:, 1] / intervals[:, 2])
    
    return start_indices, end_indices



def update_nodes_and_clusters(nodes_centers_df, clusters_list, included_nodes, excluded_nodes):
    # update nodes and cluster indices: remove nodes that dont belong to a community
    # nodes_centers_df.drop(excluded_nodes, inplace=True)
    nodes_centers_df = nodes_centers_df.loc[included_nodes]
    nodes_centers_df.reset_index(drop=True, inplace=True)

    for i, cluster in enumerate(clusters_list):
        for j, c in enumerate(cluster):
            clusters_list[i][j] = included_nodes.index(c)

    return nodes_centers_df, clusters_list


def post_disc(seq_names, matches_df, params):
    ''' modularity based clustering, as described in 
        Park&Glass, "Unsupervised pattern discovery in speech", 2008 '''

    matches_df = apply_cost_edge_w(matches_df, params['cost_thr'])
    print('*** mean edge weight is {:.3f} after cost threshold ***'.format(matches_df['edge_w'].mean()))

    similarity_profile = compute_similarity_profile(matches_df, seq_names )
    print('*** similarity profile computed ***')

    node_centers = find_node_centers(similarity_profile, peak_thr=params['peak_thr'], filter_w=13)
    print('*** node centers found ***')

    nodes_df = nodes_list_to_df(node_centers)

    edges_mat, path_idx_mat, path_idx_cnt = compute_adjacency_matrix_fast(seq_names, nodes_df, matches_df.sort_index())
    print('*** graph constructed ***')

    # edges_mat = make_symmetric(edges_mat)

    # edges_mat, path_idx_mat, path_idx_cnt, nodes_df = remove_isolated_nodes(
                            # edges_mat, path_idx_mat, path_idx_cnt, nodes_df)

    clusters_list_tmp, memberships, modularity = cluster_adj_mat(
                            edges_mat, q_thr = params['modularity_thr'], method=params['clus_alg'])
    print('*** graph clustering done !! ***')

    clusters_list, excluded_clusters, included_nodes, excluded_nodes = prune_clusters(
                            clusters_list_tmp, cluster_thr=params['min_cluster_size'])

    # start_indices, end_indices = determine_node_intervals(
                            # matches_df, nodes_df, clusters_list, path_idx_mat, path_idx_cnt, included_nodes )
    start_indices, end_indices = determine_node_intervals_new(matches_df, nodes_df, clusters_list, included_nodes)                            
    print('*** intervals found ***')

    nodes_df, clusters_list = update_nodes_and_clusters(nodes_df, clusters_list, included_nodes, excluded_nodes)

    nodes_df['start'] = start_indices
    nodes_df['end'] = end_indices

    # deduplicate clusters
    clusters_list = dedup_clusters(clusters_list, nodes_df, params['dedupthr'], params['min_cluster_size'])

    return nodes_df, clusters_list


def run_clustering_Modularity(seq_names, matches_df, params):
    
    nodes_df, clusters_list = post_disc(seq_names, matches_df, params)

    nodes_df, clusters_list = nodes_starts_from1(nodes_df, clusters_list)

    return nodes_df, clusters_list