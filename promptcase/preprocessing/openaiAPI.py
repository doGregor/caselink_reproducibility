import tiktoken
import os
from tqdm import tqdm
from openai import OpenAI

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

path = os.getcwd()
path_clean = []
for x in path.split('/'):
    if x != 'caselink_reproducibility':
        path_clean.append(x)
    else:
        path_clean.append(x)
        break
path = '/'.join(path_clean)

##########################################################################################################
## insert the openai api key

with open(path + '/openai_api_key.txt') as openai_api_key:
    api_key = openai_api_key.read()
    
client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

##########################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='coliee_2022', help="coliee_2022, coliee_2023, coliee_2024, or custom")
parser.add_argument("--data_split", default='train', type=str, help="train or test")
parser.add_argument("--llm", default='gpt-3.5-turbo-1106', type=str, help="language model name from OpenAI API")
args = parser.parse_args()

RDIR = path + '/datasets/' + args.dataset + '/' + args.data_split + '_files_processed'
WDIR = path + '/datasets/' + args.dataset + '/' + args.data_split + '_summary_txt'
files = os.listdir(RDIR)

if os.path.isdir(WDIR):
    pass
else:
    os.makedirs(WDIR)

for pfile in tqdm(files[:100]):
    if not pfile.endswith('txt'):
        continue
    file_name = pfile.split('.')[0]
    if os.path.exists(os.path.join(WDIR, file_name + '.txt')):
        pass
    else:
        with open(os.path.join(RDIR, pfile), 'r') as f:
            long_text = f.read()
            f.close()
        if len(encoding.encode(long_text)) < 500:
            summary_total = long_text
        else:
            summary_total = ''
            length = int(len(encoding.encode(long_text))/3500) + 1
            # Loop through each line in the file
            for i in range(length):
                para = long_text[3500*i:3500*(i+1)]
                for x in range(100):
                    try:
                        completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": "Summerize in 50 words:" + para,
                                }
                            ],
                            model=args.llm,
                        )
                        summary_text = completion.choices[0].message.content
                        break
                    except:
                        print('ERROR calling OpenAI API')
                        continue
                summary_total += ' ' + summary_text
        with open(os.path.join(WDIR, file_name + '.txt'), 'w') as file:
            file.write(summary_total)
            file.close()
print('finish')
    