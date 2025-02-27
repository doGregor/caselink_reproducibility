import torch
from torchmetrics import RetrievalNormalizedDCG, RetrievalMRR, RetrievalMAP, RetrievalPrecision
import json
import os


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


def t_metrics(dataset, predict_path):

    label_path = get_path() + '/datasets/' + dataset + '/test_labels.json'

    ## prediction preprocess
    pre_dic = json.load(open(predict_path, 'r'))
    label_dict = json.load(open(label_path, 'r'))

    if dataset == 'lecard':
        label_dict_origin = label_dict
        label_dict = {}
        for k, v in label_dict_origin.items():
            val = [str(value) for value in v]
            label_dict.update({k: val})

    index = -1
    index_list = []
    preds_list = []
    traget_list = []
    for key,value in pre_dic.items():
        index += 1
        rank = 1.0
        for v in value[:5]:
            index_list.append(index)
            preds_list.append(rank)
            if v in label_dict[key]:
                traget_list.append(True)
            else:
                traget_list.append(False)
            rank -= 0.05

    ## mrr@5
    mrr_index_list = []
    mrr_preds_list = []
    mrr_traget_list = []
    for key,value in pre_dic.items():
        index += 1
        rank = 1.0
        for v in value[:5]:
            mrr_index_list.append(index)
            mrr_preds_list.append(rank)
            if v in label_dict[key]:
                mrr_traget_list.append(True)
            else:
                mrr_traget_list.append(False)
            rank -= 0.05


    ndcg = RetrievalNormalizedDCG(top_k=5)

    mrr = RetrievalMRR()
    map = RetrievalMAP()
    p = RetrievalPrecision(top_k=5)
    ndcg_score = ndcg(torch.tensor(preds_list), torch.tensor(traget_list), indexes=torch.tensor(index_list))
    mrr_score = mrr(torch.tensor(mrr_preds_list), torch.tensor(mrr_traget_list), indexes=torch.tensor(mrr_index_list)) ##mrr@5
    map_score = map(torch.tensor(preds_list), torch.tensor(traget_list), indexes=torch.tensor(index_list)) 
    p_score = p(torch.tensor(preds_list), torch.tensor(traget_list), indexes=torch.tensor(index_list))

    return ndcg_score, mrr_score, map_score, p_score