#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:17:46 2019

@author: korhan
"""


import pickle
from os.path import join,dirname,abspath
import json
import pandas as pd


rootdir = dirname(abspath(__file__))

def save_obj(obj,name, path):
    with open(join(path,name+'.pkl'),'wb') as f:
        pickle.dump(obj,f,protocol=2)
        
def load_obj(name, path):
    with open(join(path,name+'.pkl'),'rb') as f:
        return pickle.load(f)

def load_obj2(path):
    with open(path,'rb') as f:
        return pickle.load(f)


def save_json(full_path,data):
    with open(full_path+'.json', 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
        
def load_json(full_path):

    if not full_path.endswith('.json'):
        full_path = full_path + '.json'
        
    with open( full_path, 'r') as fp:
        data = json.load(fp)
        return data
       

def pickle_load_nodes_clusters(postdisc_path):
    nodes_df = pd.read_pickle(join(postdisc_path,'nodes.pkl'))
    clusters_list = load_obj(name='clusters', path=postdisc_path)
    return nodes_df, clusters_list


def pickle_save_nodes_clusters(nodes_df, clusters_list, postdisc_path):
    nodes_df.to_pickle(join(postdisc_path,'nodes.pkl'), protocol=3)
    save_obj(name='clusters', path=postdisc_path, obj=clusters_list)
        


        