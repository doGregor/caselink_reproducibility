import os
import subprocess
import glob
import pandas as pd
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='coliee_2022', help="coliee_2022, coliee_2023, coliee_2024, or custom")
parser.add_argument("--data_split", default='train', type=str, help="train or test")
parser.add_argument("--feature", type=str, default='issue', help="fact or issue")
parser.add_argument('--llm', type=str, default='gpt', help="gpt or llama")
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

if args.feature == "fact":
    if args.llm == "gpt":
        input_path = path + '/datasets/' + args.dataset + '/information_extraction/' + args.data_split + '_summary'
    elif args.llm == 'llama':
        input_path = path + '/datasets/' + args.dataset + '/information_extraction/' + args.data_split + '_summary_llama'
    else:
        sys.exit("No valid LLM")
else:
    input_path = path + '/datasets/' + args.dataset + '/information_extraction/' + args.data_split + '_referenced'


def Stanford_Relation_Extractor():
    
    print('Relation Extraction Started')
    for f in tqdm(glob.glob(input_path+"/kg/*.txt")):
        if os.path.exists(f + '-out.csv'):
            continue
        else:
            os.chdir(path + '/casegnn/information_extraction/stanford-openie')

            p = subprocess.Popen(['./process_large_corpus.sh',f,f + '-out.csv'], stdout=subprocess.PIPE)
            output, err = p.communicate()
            
            os.chdir( '../..')
   

    print('Relation Extraction Completed')


if __name__ == '__main__':
    Stanford_Relation_Extractor()
