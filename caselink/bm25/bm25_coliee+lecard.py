"""
Credits: this file is based on the original CaseLink code by @yanran-tang
"""
import os
import sys
sys.path.append('.')
sys.path.append('..')

import tqdm
from tqdm import tqdm
import json
import torch

from bm25_model import BM25

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## Training config
parser.add_argument("--ngram_1", type=int, default=3,
                    help="ngram")
parser.add_argument("--ngram_2", type=int, default=4,
                    help="ngram")
parser.add_argument("--dataset", type=str, default='coliee_2022', help="coliee_2022, coliee_2023, coliee_2024, or custom")
parser.add_argument("--data_split", default='train', type=str, help="train or test")
args = parser.parse_args()

print(args)

path = os.getcwd()
path_clean = []
for x in path.split('/'):
    if x != 'caselink_reproducibility':
        path_clean.append(x)
    else:
        path_clean.append(x)
        break
path = '/'.join(path_clean)
RDIR = path+'/datasets/'+args.dataset+'/'+args.data_split+'_files'
files = os.listdir(RDIR)

print('check if bm25 matrix already exists...')
dataset_folder_path = path + '/datasets/' + args.dataset
if os.path.isdir(dataset_folder_path + '/bm25'):
    pass
else:
    os.makedirs(dataset_folder_path + '/bm25')

# bm25 computing starts    
corpus =[]
corpus_sequence_names = []
corpus_dict = {}

print('Corpus loading: ')
for pfile in tqdm(files):
    if not pfile.endswith('txt'):
        continue
    file_name = pfile
    with open(os.path.join(RDIR, pfile), 'r') as f:
        original_text = f.read()
        f.close()

    text = original_text

    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('"', '')
    text = text.replace(',', '')
    text = text.replace('\\', '')

    txt = text
    corpus.append(txt)
    corpus_dict[file_name] = txt
    corpus_sequence_names.append(file_name)

bm25 = BM25(ngram_range=(args.ngram_1, args.ngram_2))
bm25.fit(corpus)

score_dict = {}
prediction_dict = {}
final_prediction_dict = {}
print('BM25 calculation start: ')
for i in tqdm(range(len(corpus_sequence_names))):
    if i == 0:
        query_name = corpus_sequence_names[i]
        print(query_name)
        que_text = corpus_dict[query_name]
        doc_scores = bm25.transform(que_text, corpus)
        bm25_matrix = torch.unsqueeze(torch.FloatTensor(doc_scores), 0)
    else:
        query_name = corpus_sequence_names[i]
        print(query_name)
        que_text = corpus_dict[query_name]
        doc_scores = bm25.transform(que_text, corpus)
        score_tensor = torch.unsqueeze(torch.FloatTensor(doc_scores), 0)         
        bm25_matrix = torch.cat((bm25_matrix, score_tensor), 0) 
        
torch.save(bm25_matrix, dataset_folder_path + '/bm25/' +  args.dataset + '_' + args.data_split + '_BM25_score_matrix.pt')
with open (dataset_folder_path + '/bm25/' +  args.dataset + '_' + args.data_split + '_BM25_score_matrix_case_sequence.json', 'w') as f:
    json.dump(corpus_sequence_names, f)
    f.close()

print('Finished')
