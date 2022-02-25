
from os.path import join
import os
import sys
sys.path.append('../')

from clustering.pairwise import run_clustering_pairs
# from clustering.comm_detection import run_clustering_Modularity
from utils.helper_fncs import pickle_load_nodes_clusters, pickle_save_nodes_clusters



def gen_postdisc_name(params):
    """ generate name for clustering experiment (required for bookkeeping)"""

    if params['method'] == 'modularity':
        postdisc_name = 'post_cost{}_peak{}_q{}_{}Alg_mc{}'.format(
                params['cost_thr'], params['peak_thr'], params['modularity_thr'],
                params['clus_alg'], params['min_cluster_size'])

    # if params['method'] == 'zr17':
    #     postdisc_name = 'postZR_cost{}_olap{}_dedup{}_edw{}_dtw{}'.format(
    #             params['cost_thr'], params['olapthr'], 
    #             params['dedupthr'], params['min_ew'], params['dtwth'])

    # if params['method'] == 'custom':
    #     postdisc_name = 'post_customclus_cost{}_{}Alg_dedup{}_mix{}'.format(
    #             params['cost_thr'], params['clus_alg'], 
    #             params['dedupthr'], params['mix_ratio'])

    if params['method'] == 'pairwise':
        postdisc_name = 'postpairwise_cost{}_olap{}'.format(
                params['cost_thr'], params['olapthr_m'] )

    return postdisc_name



def run_clustering(seq_names, matches_df, params):
    # runs the clustering part
       
    postdisc_name = gen_postdisc_name(params['clustering'])
    
    postdisc_path = join(params['exp_root'], params['expname'], postdisc_name)

    if (os.path.exists(join(postdisc_path,'nodes.pkl')) and      
        os.path.exists(join(postdisc_path,'clusters.pkl'))):
         
        nodes_df, clusters_list = pickle_load_nodes_clusters(postdisc_path)

    else: #  if not computed before

        # if params['clustering']['method'] == 'modularity':

        #     os.makedirs(postdisc_path, exist_ok=True)
        #     nodes_df, clusters_list = run_clustering_Modularity(
        #         seq_names, matches_df, params['clustering'])

        # if params['clustering']['method'] == 'zr17':

        #     nodes_df, clusters_list = run_clustering_ZR(
        #         matches_df, params, postdisc_path, postdisc_name)

        # if params['clustering']['method'] == 'custom':

        #     os.makedirs(postdisc_path, exist_ok=True)
        #     nodes_df, clusters_list = run_custom_clustering(
        #         matches_df, params['clustering'])

        if params['clustering']['method'] == 'pairwise':

            os.makedirs(postdisc_path, exist_ok=True)
            nodes_df, clusters_list = run_clustering_pairs(
                matches_df, params['clustering'])


        pickle_save_nodes_clusters(nodes_df, clusters_list, postdisc_path)        
        
    return nodes_df, clusters_list, postdisc_name
