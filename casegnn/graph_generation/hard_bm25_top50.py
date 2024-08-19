import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='coliee_2022', help="coliee_2022, coliee_2023, coliee_2024, or custom")
args = parser.parse_args()


path = os.getcwd()
path_clean = []
for x in path.split('/'):
    if x != 'caselink_reproducibility':
        path_clean.append(x)
    else:
        path_clean.append(x)
        break
path = '/'.join(path_clean)


path_to_dataset = path + '/datasets/' + args.dataset

# # #

with open(path_to_dataset + '/bm25/' + args.dataset + '_train_BM25_score_matrix_case_sequence.json') as f:
    bm25_list = json.load(f)

with open(path_to_dataset + '/bm25/' + args.dataset + '_train_BM25_score_matrix.pt', "rb") as fIn:
    bm25_score_matrix = torch.load(fIn)
    
# # #    

with open(path_to_dataset + '/train_labels.json') as f:
    train_labels = json.load(f)
    
# query_cases = train_labels.keys()
# candidate_cases = np.array(list(set(bm25_list) - set(query_cases)))

# # #

output_dicts = []
for case_id in tqdm(list(train_labels.keys())):
    
    exlusion_list = train_labels[case_id] + [case_id]
    boolean_mask = []
    for c_id in bm25_list:
        if c_id in exlusion_list:
            boolean_mask.append(True)
        else:
            boolean_mask.append(False)
    
    single_dict = {}
    idx_bm25_case = bm25_list.index(case_id)
    
    case_row = np.array(bm25_score_matrix[idx_bm25_case,:])
    case_row[np.array(boolean_mask)] = 0
    
    top_50_indices = np.argsort(case_row)[-50:][::-1]
    x = np.array(bm25_list)[top_50_indices]
    
    single_dict[case_id] = x.tolist()
    output_dicts.append(f'{single_dict}\n')
    
with open(path_to_dataset + '/hard_neg_top50_train.json', 'w') as out_file:
    out_file.writelines(output_dicts)
