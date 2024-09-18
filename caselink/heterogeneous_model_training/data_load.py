import dgl
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset
from torch_geometric.data import InMemoryDataset
import random
from torch.utils.data import Dataset

import json
import torch

import os
    
    
class HeteroSyntheticDataset(InMemoryDataset):
    def __init__(self, file_path, label_dict, train_pool, hard_neg_num, hard_bm25_dict):
        self.graph_and_label = torch.load(file_path)
        self.label_dict = label_dict
        self.hard_bm25_dict = hard_bm25_dict
        self.train_pool = train_pool
        self.hard_neg_num = hard_neg_num

        super(HeteroSyntheticDataset, self).__init__(root='/home/jovyan/tmp')
    
    @property
    def raw_file_names(self):
        return ['raw_data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    """
    def download(self):
        pass
    """
    
    def process(self):
        self.graph_list = self.graph_and_label[0]
        self.label_list = self.graph_and_label[1]['case_name_list']
        self.label_list = [str(int(self.label_list[x])).zfill(6) for x in range(len(self.label_list))]
        self.case_index = self.graph_and_label[2]

        label_dict = self.label_dict
        hard_bm25_dict = self.hard_bm25_dict
        train_pool = self.train_pool
        hard_neg_num = self.hard_neg_num
        pos_case_list = []
        ran_neg_list = []
        hard_neg_list = []
        query_index_list = []
        for x in range(len(self.label_list)):
            query_name = self.label_list[x]+'.txt'
            pos_case = random.choice(label_dict[query_name]).split('.')[0]
            pos_case_index = train_pool.index(pos_case)
            pos_case_list.append(pos_case_index)
            query_index_list.append(train_pool.index(self.label_list[x]))

            i = 0
            while i<4400: 
                ran_neg_case = random.choice(train_pool)
                if ran_neg_case+'.txt' not in label_dict[query_name]:
                    break
                break
            
            ran_neg_case_index = train_pool.index(ran_neg_case)
            ran_neg_list.append(ran_neg_case_index)
            
            hard_neg_sublist = []
            for i in range(hard_neg_num):
                bm25_neg_case = random.choice(hard_bm25_dict[query_name]).split('.')[0]
                bm25_neg_case_index = train_pool.index(bm25_neg_case)
                hard_neg_sublist.append(bm25_neg_case_index)  
            hard_neg_list.append(hard_neg_sublist)
               
        self.pos_case_list = pos_case_list
        self.ran_neg_list = ran_neg_list
        self.hard_neg_list = hard_neg_list
        self.query_index_list = query_index_list
    
    def __getitem__(self, i):
        return self.query_index_list[i], self.label_list[i], self.pos_case_list[i], self.ran_neg_list[i], self.hard_neg_list[i]

    def __len__(self):
        return len(self.graph_list)



def collate(samples):
    query_index, labels, pos_case, ran_neg, hard_neg = map(list, zip(*samples))
            
    return query_index, labels, pos_case, ran_neg, hard_neg
