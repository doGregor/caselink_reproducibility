import os
from tqdm import tqdm
import transformers
import torch


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
LLM = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = transformers.AutoTokenizer.from_pretrained(LLM)

pipeline = transformers.pipeline(
    "text-generation",
    model=LLM,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

##########################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='coliee_2022', help="coliee_2022, coliee_2023, coliee_2024, or custom")
parser.add_argument("--data_split", default='train', type=str, help="train or test")
args = parser.parse_args()

RDIR = path + '/datasets/' + args.dataset + '/' + args.data_split + '_files_processed'
WDIR = path + '/datasets/' + args.dataset + '/' + args.data_split + '_summary_txt_llama'
files = os.listdir(RDIR)

if os.path.isdir(WDIR):
    pass
else:
    os.makedirs(WDIR)

for pfile in tqdm(files[:]):
    if not pfile.endswith('txt'):
        continue
    file_name = pfile.split('.')[0]
    if os.path.exists(os.path.join(WDIR, file_name + '.txt')):
        pass
    else:
        with open(os.path.join(RDIR, pfile), 'r') as f:
            long_text = f.read()
            f.close()
        if len(tokenizer.tokenize(long_text)) < 500:
            summary_total = long_text
        else:
            summary_total = ''
            length = int(len(tokenizer.tokenize(long_text))/3500) + 1
            # Loop through each line in the file
            for i in range(length):
                para = long_text[3500*i:3500*(i+1)]
                for x in range(100):
                    try:
                        messages = [
                            {"role": "user", "content": "Summarize in 50 words:" + para},
                        ]
                        outputs = pipeline(
                            messages,
                            max_new_tokens=256,
                            pad_token_id=pipeline.tokenizer.eos_token_id
                        )
                        summary_text = outputs[0]["generated_text"][-1]['content']
                        break
                    except:
                        print('ERROR calling Llama')
                        continue
                summary_total += ' ' + summary_text
        with open(os.path.join(WDIR, file_name + '.txt'), 'w') as file:
            file.write(summary_total)
            file.close()
print('finish')