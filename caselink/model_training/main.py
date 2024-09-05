import random
import torch.nn as nn
import json
import os
from tqdm import tqdm

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from dgl.data.utils import load_graphs
from data_load import SyntheticDataset, collate
from CaseLink_model import CaseLink, early_stopping

from train import forward

from torch.utils.tensorboard import SummaryWriter
import time
import logging

import argparse
parser = argparse.ArgumentParser()
## model parameters
parser.add_argument("--in_dim", type=int, default=1536, help="input_feature_dimension")
parser.add_argument("--h_dim", type=int, default=1536, help="hidden_feature_dimension")
parser.add_argument("--out_dim", type=int, default=1536, help="output_feature_dimension")                                
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout for all subsequent layers")

## training parameters
parser.add_argument("--epoch", type=int, default=100, help="training epochs")
parser.add_argument("--lr", type=float, default=1e-04, help="learning rate")
parser.add_argument("--wd", default=1e-06, type=float, help="Weight decay if we apply some.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--temp", type=float, default=0.1, help="Temperature for relu")
parser.add_argument("--hard_neg_num", type=int, default=5, help="bm25_neg case number")
parser.add_argument("--num_heads", type=int, default=1, help="Numbers of attention heads")
parser.add_argument("--ran_neg_num", type=int, default=1, help="random_neg case number")
parser.add_argument("--layer_num", type=int, default=2, help="numbers of GNN layers")
parser.add_argument("--topk_neighbor", type=int, default=5, help="5 10 20")
parser.add_argument("--charge_threshold", type=float, default=0.9, help="0.85 0.9 0.95") 
parser.add_argument("--lamb", type=float, default=0.001, help="lambda of l_reg") 

## other parameters
parser.add_argument("--dataset", type=str, default='coliee_2022', help="coliee_2022, coliee_2023, coliee_2024, or custom")
parser.add_argument("--gpu", type=str, default='0', help="CUDA GPU position")

args = parser.parse_args()

## Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')
logging.warning(args)


def get_path():
    path = os.getcwd()
    path_clean = []
    for x in path.split('/'):
        if x != 'caselink_reproducibility':
            path_clean.append(x)
        else:
            path_clean.append(x)
            break
    path = '/'.join(path_clean)
    return path


def main():
    log_dir = get_path() + '/datasets/' + args.dataset + '/caselink_experiments/'
    training_setup = 'lamb' + str(args.lamb) + '_neibtop' + str(args.topk_neighbor) + "_charthre" + str(args.charge_threshold) + '_' + str(args.layer_num) + 'layer_bs' + str(args.batch_size) + '_dp' + str(args.dropout) + '_lr' + str(args.lr) + '_wd'+str(args.wd) + '_t' + str(args.temp) + '_headnum' + str(args.num_heads) + '_hardneg' + str(args.hard_neg_num) + '_ranneg' + str(args.ran_neg_num) + '_' + time.strftime("%m%d-%H%M%S")
    log_dir += training_setup

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    print(log_dir)

    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    ## model initialization
    model = CaseLink(args.in_dim, args.h_dim, args.out_dim, dropout=args.dropout, layer_num=args.layer_num, num_heads=args.num_heads)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    ## Load train label
    train_labels = {}
    with open(get_path() + '/datasets/' + args.dataset + '/train_labels.json', 'r')as f:
        train_labels = json.load(f)
        f.close() 

    bm25_hard_neg_dict = {}
    with open(get_path() + '/datasets/' + args.dataset + '/hard_neg_top50_train.json', 'r')as file:
        for line in file.readlines():
            line = line.replace("\'", "\"")
            dic = json.loads(line)
            bm25_hard_neg_dict.update(dic)
        file.close() 

    ## Load test label
    test_labels = {}
    with open(get_path() + '/datasets/' + args.dataset + '/test_labels.json', 'r')as f:
        test_labels = json.load(f)
        f.close()    

    yf_path = get_path() + '/datasets/' + args.dataset + '/test_candidate_with_yearfilter.json'

    ## Load datasets
    ## Train dataset
    train_graph_and_label = load_graphs(get_path() + '/datasets/' + args.dataset + '/graphs/caselink/train_bm25top' + str(args.topk_neighbor) + '_charge_thres' + str(args.charge_threshold) + '.bin')
    train_graphs = train_graph_and_label[0]
    graph_labels = train_graph_and_label[1]['case_name_list'].tolist()
    train_label_list = [str(int(x)).zfill(6) for x in graph_labels]
    
    train_dataset = SyntheticDataset(file_path=get_path() + '/datasets/' + args.dataset + '/graphs/caselink/train_graph_bin_bm25top' + str(args.topk_neighbor) + '_charge_thres' + str(args.charge_threshold) + '.pt', label_dict=train_labels, train_pool=train_label_list, hard_neg_num=args.hard_neg_num, hard_bm25_dict=bm25_hard_neg_dict)
    train_graph = train_dataset.graph_list
    train_label = train_dataset.label_list
    train_sampler = SubsetRandomSampler(torch.arange(len(train_graph)))
    train_dataloader = GraphDataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=False, collate_fn=collate)
    
    train_candidate_list = []
    for i in train_label_list:
        if i+'.txt' not in train_labels.keys():
            train_candidate_list.append(train_label_list.index(i))
    
    ## Test dataset 
    test_dataloader = None       
    test_graph_and_label = load_graphs(get_path() + '/datasets/' + args.dataset + '/graphs/caselink/test_bm25top' + str(args.topk_neighbor) + '_charge_thres' + str(args.charge_threshold) + '.bin')
    test_graphs = test_graph_and_label[0]
    test_graph_labels = test_graph_and_label[1]['case_name_list'].tolist()
    test_label_list = [str(int(x)).zfill(6) for x in test_graph_labels]    
    
    test_mask = []
    test_query_list = []
    test_query_index_list = []
    for k,v in test_labels.items():
        test_mask_0 = []
        test_query_list.append(k)
        case_index = test_label_list.index(k.split('.')[0])
        test_query_index_list.append(case_index)
        for i in range(len(test_label_list)):
            case = test_label_list[i]+'.txt'
            if case in v:
                test_mask_0.append(1)
            else:
                test_mask_0.append(0)  
        test_mask.append(torch.FloatTensor(test_mask_0))
    test_mask = torch.stack(test_mask).to(device)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))

    highest_ndcg = 0
    con_epoch_num = 0    
    for epoch in tqdm(range(args.epoch)):
        print('Epoch:', epoch)
        forward(args.dataset, model, device, writer, train_dataloader, train_graph_and_label, train_labels, train_candidate_list, yf_path, epoch, args.batch_size, args.temp, args.lamb, args.hard_neg_num, train_flag=True, test_mask=None, test_label_list=None, test_query_list=None, test_query_index_list=None, optimizer=optimizer, training_setup=training_setup)
        with torch.no_grad():            
            ndcg_score_yf = forward(args.dataset, model, device, writer, test_dataloader, test_graph_and_label, test_labels, train_candidate_list, yf_path, epoch, args.batch_size, args.temp, args.lamb, args.hard_neg_num, train_flag=False, test_mask=test_mask, test_label_list=test_label_list, test_query_list=test_query_list, test_query_index_list=test_query_index_list, optimizer=optimizer, training_setup=training_setup)

        ## Early stopping
        stop_para = early_stopping(highest_ndcg, ndcg_score_yf, epoch, con_epoch_num)
        highest_ndcg = stop_para[0]
        if stop_para[1]:
            break
        else:
            con_epoch_num = stop_para[2]

if __name__ == '__main__':
    main()

