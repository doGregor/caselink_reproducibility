import torch
import torch.nn.functional as F
import json
import os
from tqdm import tqdm

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data.utils import load_graphs, save_graphs

from data_load import SyntheticDataset, PoolDataset, collate
from model import CaseGNN, early_stopping

from train import forward

from torch.utils.tensorboard import SummaryWriter
import time
import logging

import argparse
parser = argparse.ArgumentParser()
## model parameters
parser.add_argument("--in_dim", type=int, default=768, help="Input_feature_dimension")
parser.add_argument("--h_dim", type=int, default=768, help="Hidden_feature_dimension")
parser.add_argument("--out_dim", type=int, default=768, help="Output_feature_dimension")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout for embedding / GNN layer ")       
parser.add_argument("--num_head", default=1, type=int, help="Head number of GNN layer ")                            

## training parameters
parser.add_argument("--epoch", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate")
parser.add_argument("--wd", default=1e-05, type=float, help="Weight decay if we apply some.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--temp", type=float, default=0.1, help="Temperature for relu")
parser.add_argument("--ran_neg_num", type=int, default=1, help="Random sampled case number")
parser.add_argument("--hard_neg", type=bool, default=True, help="Using bm25_neg or not")
parser.add_argument("--hard_neg_num", type=int, default=1, help="Bm25_neg case number")

## other parameters
parser.add_argument("--dataset", type=str, default='coliee_2022', help="coliee_2022, coliee_2023, coliee_2024, or custom")
parser.add_argument('--llm', type=str, default='gpt', help="gpt or llama")
args = parser.parse_args()

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')
logging.warning(args)

path = os.getcwd()
path_clean = []
for x in path.split('/'):
    if x != 'caselink_reproducibility':
        path_clean.append(x)
    else:
        path_clean.append(x)
        break
path = '/'.join(path_clean)

def main():
    if args.llm == 'gpt':
        suffix = ''
    elif args.llm == 'llama':
        suffix = '_llama'
    else:
        sys.exit("No valid LLM") 
    
    log_dir = path + '/datasets/' + args.dataset + '/casegnn_experiments' + suffix + '/'
    training_setup = 'bs' + str(args.batch_size) + '_dp' + str(args.dropout) + '_lr' + str(args.lr) + '_wd' + str(args.wd) + '_t' + str(args.temp) + '_headnum' + str(args.num_head) + '_hardneg' + str(args.hard_neg_num) + '_ranneg' + str(args.ran_neg_num) + '_' + time.strftime("%m%d-%H%M%S")
    log_dir += training_setup

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## model initialization
    model = CaseGNN(args.in_dim, args.h_dim, args.out_dim, dropout=args.dropout, num_head=args.num_head)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    ## Dataset initialization
    
    # Train dataset
    train_dataset = SyntheticDataset(path + '/datasets/' + args.dataset + '/graphs' + suffix + '/casegnn/train_fact_graph_synthetic.bin')
    train_graph = train_dataset.graph_list
    train_label = train_dataset.label_list
    train_sampler = SubsetRandomSampler(torch.arange(len(train_graph)))
    train_dataloader = GraphDataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=False, collate_fn=collate)

    train_sumfact_pool_dataset = PoolDataset(path + '/datasets/' + args.dataset + '/graphs' + suffix + '/casegnn/train_fact_graph.bin')
    train_referissue_pool_dataset = PoolDataset(path + '/datasets/' + args.dataset + '/graphs/casegnn/train_issue_graph.bin')
    
    # Test dataset
    ##Inference batch size
    num_test_files = [file for file in os.listdir(path + '/datasets/' + args.dataset + '/test_files') if file.endswith('txt')]
    inference_bs = len(num_test_files)
        
    test_sumfact_dataset = SyntheticDataset(path + '/datasets/' + args.dataset + '/graphs' + suffix + '/casegnn/test_fact_graph_synthetic.bin')

    test_sumfact_graph = test_sumfact_dataset.graph_list
    test_sumfact_sampler = SubsetRandomSampler(torch.arange(len(test_sumfact_graph)))
    test_dataloader = GraphDataLoader(
        test_sumfact_dataset, sampler=test_sumfact_sampler, batch_size=inference_bs, drop_last=False, collate_fn=collate, shuffle=False)

    test_sumfact_pool_dataset = PoolDataset(path + '/datasets/' + args.dataset + '/graphs' + suffix + '/casegnn/test_fact_graph.bin')
    test_referissue_pool_dataset = PoolDataset(path + '/datasets/' + args.dataset + '/graphs/casegnn/test_issue_graph.bin')

    ## load train label
    train_labels = {}
    with open(path + '/datasets/' + args.dataset + '/train_labels.json', 'r')as f:
        train_labels = json.load(f)
        f.close() 

    bm25_hard_neg_dict = {}
    with open(path + '/datasets/' + args.dataset + '/hard_neg_top50_train.json', 'r')as file:
        for line in file.readlines():
            line = line.replace("\'", "\"")
            dic = json.loads(line)
            bm25_hard_neg_dict.update(dic)
        file.close() 

    # ## load test label
    test_labels = {}
    with open(path + '/datasets/' + args.dataset + '/test_labels.json', 'r')as f:
        test_labels = json.load(f)
        f.close()    

    yf_path = path + '/datasets/' + args.dataset + '/test_candidate_with_yearfilter.json' 

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))

    highest_ndcg_score_yf = 0
    highest_ndcg = 0
    con_epoch_num = 0
    for epoch in tqdm(range(args.epoch)):
        print('Epoch:', epoch)
        forward(args.dataset, model, device, writer, train_dataloader, train_sumfact_pool_dataset, train_referissue_pool_dataset, train_labels, yf_path, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, train_flag=True, embedding_saving=False, prediction_saving=False, optimizer=optimizer, training_setup=training_setup, suffix=suffix)
        with torch.no_grad():                      
            ndcg_score_yf = forward(args.dataset, model, device, writer, test_dataloader, test_sumfact_pool_dataset, test_referissue_pool_dataset, test_labels, yf_path, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, train_flag=False, embedding_saving=False, prediction_saving=False, optimizer=optimizer, training_setup=training_setup, suffix=suffix)
        if ndcg_score_yf > highest_ndcg_score_yf:
            highest_ndcg_score_yf = ndcg_score_yf
            ndcg_score_yf = forward(args.dataset, model, device, writer, test_dataloader, test_sumfact_pool_dataset, test_referissue_pool_dataset, test_labels, yf_path, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, train_flag=False, embedding_saving=False, prediction_saving=True, optimizer=optimizer, training_setup=training_setup, suffix=suffix)

        stop_para = early_stopping(highest_ndcg, ndcg_score_yf, epoch, con_epoch_num)
        highest_ndcg = stop_para[0]
        if stop_para[1]:
            break
        else:
            con_epoch_num = stop_para[2]
    ##CaseGNN Embedding Saving
    forward(args.dataset, model, device, writer, train_dataloader, train_sumfact_pool_dataset, train_referissue_pool_dataset, train_labels, yf_path, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, train_flag=True, embedding_saving=True, prediction_saving=False, optimizer=optimizer, training_setup=training_setup, suffix=suffix)
    forward(args.dataset, model, device, writer, test_dataloader, test_sumfact_pool_dataset, test_referissue_pool_dataset, test_labels, yf_path, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, train_flag=False, embedding_saving=True, prediction_saving=False, optimizer=optimizer, training_setup=training_setup, suffix=suffix)

if __name__ == '__main__':
    main()