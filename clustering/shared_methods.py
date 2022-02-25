
import numpy as np
import pandas as pd

from utils.overlaps import compute_overlap


def apply_cost_edge_w(matches_df, cost_thr=0.1):
    """ Apply cost threshold to normalized edge weights,

    Args:
        matches_df (pandas.DataFrame): Pairs of similar segments, 
            together with dissimilarity score 'cost' column
        cost_thr (float, optional): Ratio of best matches to retain. Defaults to 0.1.

    Returns:
        pandas.DataFrame: Thresholded matches.
    """    
    
    cost_max = matches_df.cost.max()
    # convert to normalized similarity scores
    matches_df['edge_w'] = matches_df['cost'].apply(lambda x: (cost_max - x) / cost_max)
    
    cost_val = matches_df['edge_w'].sort_values(ascending=False).values[round(len(matches_df) * cost_thr)]
    matches_df = matches_df.loc[matches_df['edge_w'] > cost_val].copy()
    matches_df.reset_index(inplace=True, drop=True)

    return matches_df


def nodes_starts_from1(nodes_df, clusters_list):
    # ensure the incides start from at least 1
    nodes_df.index += 1
    for c,clus in enumerate(clusters_list):
        clusters_list[c] = list(np.array(clus)+1)

    return nodes_df, clusters_list

    
def prune_clusters(clusters_list_tmp, cluster_thr=2):
    """ 
    need to remove singletons after clustering is performed, 
    can also be thresholded to limit min cluster size
    """

    clusters_list = []
    excluded_clusters = []
    excluded_nodes = []
    all_nodes = set()

    for i, cluster in enumerate(clusters_list_tmp):

        all_nodes |= set(cluster)

        if len(cluster) < cluster_thr:
            excluded_clusters.append(i)
            excluded_nodes.extend(cluster)
        else:
            clusters_list.append(cluster)

    included_nodes = sorted( list(all_nodes - set(excluded_nodes)) )

    return clusters_list, excluded_clusters, included_nodes, excluded_nodes


def dedup_clusters(clusters_list, nodes_df, dedupthr=0.5, minclussize=2):
    """ deduplicate inter-cluster segments, 
        remove segment if overlaps more than 'dedupthr' with another 
        segment from the same cluster """

    for c,cluster in enumerate(clusters_list):
        n = 0
        while n < len(cluster):

            to_remove = []
            i1 = int(cluster[n])

            for m in range(n+1,len(cluster)):
                i2 = int(cluster[m])

                if nodes_df.filename[i1] == nodes_df.filename[i2]:

                    xA = nodes_df.start[i1]
                    xB = nodes_df.end[i1]
                    yA = nodes_df.start[i2]
                    yB = nodes_df.end[i2]

                    folap = compute_overlap(xA, xB, yA, yB)

                    if folap >= dedupthr:
                        to_remove.append(i2)
            n += 1

            to_remove.reverse()
            for i2 in to_remove:
                cluster.remove(i2)

            if len(cluster) >= minclussize:
                clusters_list[c] = cluster
            else:
                del clusters_list[c]
                
    return clusters_list
