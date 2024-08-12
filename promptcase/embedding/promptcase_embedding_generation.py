import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel

import json
import os
from tqdm import tqdm
import sys
sys.path.append('.')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='coliee_2022', help="coliee_2022, coliee_2023, coliee_2024, or custom")
parser.add_argument("--data_split", default='train', type=str, help="train or test")
args = parser.parse_args()

model_name = 'CSHaitao/SAILER_en_finetune'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

path = os.getcwd()
path_clean = []
for x in path.split('/'):
    if x != 'caselink_reproducibility':
        path_clean.append(x)
    else:
        path_clean.append(x)
        break
path = '/'.join(path_clean)

RDIR_sum = path + '/datasets/' + args.dataset + '/information_extraction/' + args.data_split + '_summary/result/'
RDIR_refer_sen = path + '/datasets/' + args.dataset + '/information_extraction/' + args.data_split + '_referenced/result/'
files = os.listdir(RDIR_sum)


## case representation calculation
label_list = []
model.eval()
num = 0
embedding_list = []
with torch.no_grad():
    embedding_dict = {}

    for pfile in tqdm(files[:]):
        num += 1
        file_name = pfile.split('.')[0]+'.txt'
        label_list.append(file_name)
        with open(os.path.join(RDIR_sum, pfile), 'r') as f:
            original_sum_text = f.read()
            f.close()
        with open(os.path.join(RDIR_refer_sen, pfile), 'r') as file:
            original_refer_text = file.read()
            file.close()
        fact_text = "Legal facts:"+original_sum_text
        issue_text = "Legal issues:"+original_refer_text        
        cross_text = fact_text+' '+issue_text

        if num == 1:
            ## dual encoding
            fact_tokenized_id = tokenizer(fact_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            fact_tokenized_id.to(device)
            fact_embedding = model(**fact_tokenized_id)
            fact_embedding_matrix = fact_embedding[0][:,0] ##cls token embedding [1,768]

            issue_tokenized_id = tokenizer(issue_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            issue_tokenized_id.to(device)
            issue_embedding = model(**issue_tokenized_id)
            issue_embedding_matrix = issue_embedding[0][:,0] ##cls token embedding [1,768]             
            
            ## cross encoding
            cross_tokenized_id = tokenizer(cross_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            cross_tokenized_id.to(device)
            cross_embedding = model(**cross_tokenized_id)
            cross_embedding_matrix = cross_embedding[0][:,0] ##cls token embedding [1,768]                           
        
        else:
            fact_tokenized_id = tokenizer(fact_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            fact_tokenized_id.to(device)
            fact_embedding = model(**fact_tokenized_id)
            fact_cls_embedding = fact_embedding[0][:,0] ##cls token embedding [1,768]   
            fact_embedding_matrix = torch.cat((fact_embedding_matrix, fact_cls_embedding), 0)               
            
            issue_tokenized_id = tokenizer(issue_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            issue_tokenized_id.to(device)
            issue_embedding = model(**issue_tokenized_id)
            issue_cls_embedding = issue_embedding[0][:,0] ##cls token embedding [1,768]
            issue_embedding_matrix = torch.cat((issue_embedding_matrix,issue_cls_embedding), 0)

            cross_tokenized_id = tokenizer(cross_text, return_tensors="pt", padding=False, truncation=True, max_length=512)
            cross_tokenized_id.to(device)
            cross_embedding = model(**cross_tokenized_id)
            cross_cls_embedding = cross_embedding[0][:,0] ##cls token embedding [1,768]    
            cross_embedding_matrix = torch.cat((cross_embedding_matrix, cross_cls_embedding), 0) 

WDIR = path + '/datasets/' + args.dataset + '/promptcase_embeddings'
if os.path.isdir(WDIR):
    pass
else:
    os.makedirs(WDIR)
fact_embedding_matrix = fact_embedding_matrix.detach().cpu()
issue_embedding_matrix = issue_embedding_matrix.detach().cpu()
cross_embedding_matrix = cross_embedding_matrix.detach().cpu()
embedding_list.append([fact_embedding_matrix, issue_embedding_matrix, cross_embedding_matrix])
torch.save(embedding_list, WDIR + '/' + args.data_split + '_fact_issue_cross_embedding.pt')

with open(WDIR + '/' + args.data_split + '_fact_issue_cross_embedding_case_list.json' , "w") as fOut:
    json.dump(label_list, fOut)
    fOut.close()             

print('PromptCase embedding generation finished.')