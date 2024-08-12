import re
import os
import json
import argparse


pattern = r'\b(18\d{2}|19\d{2}|200\d|201\d|202[0-3])\b'

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

label_file_path = path + '/datasets/' + args.dataset + '/test_labels.json'
output_file_path_no_labels = path + '/datasets/' + args.dataset + '/test_no_labels.json'
output_file_path_candidate = path + '/datasets/' + args.dataset + '/test_candidate_with_yearfilter.json'

with open(label_file_path) as f:
    a = json.load(f)
    
'''
with open(output_file_path_no_labels, "w+") as f:
    json.dump(a.keys(), f)
'''

names = os.listdir(path + '/datasets/' + args.dataset + '/test_files')
names = [q for q in names if q.endswith('txt')]
yd = {}
for q in names:
    with open(path + '/datasets/' + args.dataset + f'/test_files/{q}', "r", encoding="utf-8") as f:
        txt = f.read()
    years = re.findall(pattern, txt)
    years = [int(y) for y in years]
    yd[q] = max(years, default=0)

less = 0
more = 0

dedicate = {}

for q in a.keys():
    if yd[q] == 0:
        dedicate[q] = names
        continue
    dedicate[q] = []
    for doc in names:
        if yd[doc] <= yd[q]:
            dedicate[q].append(doc)

with open(output_file_path_candidate, "w+") as f:
    json.dump(dedicate, f)

print(len(dedicate))
print(sum(len(dedicate[q]) for q in dedicate) // len(dedicate))
print(len(names))
