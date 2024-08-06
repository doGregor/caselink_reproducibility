import sys
import pickle
import os
import glob
from tqdm import tqdm

import en_core_web_trf
# import en_core_web_sm

from lexnlp.extract.en import acts 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='coliee_2022', help="coliee_2022, coliee_2023, coliee_2024, or custom")
parser.add_argument("--data_split", default='train', type=str, help="train or test")
parser.add_argument("--feature", type=str, default='issue', help="fact or issue")
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
    input_path = path + '/datasets/' + args.dataset + '/' + args.data_split + '_summary_txt'
    output_path = path + '/datasets/' + args.dataset + '/information_extraction/' + args.data_split + '_summary'
else:
    input_path = path + '/datasets/' + args.dataset + '/' + args.data_split + '_files_processed_and_referenced'
    output_path = path + '/datasets/' + args.dataset + '/information_extraction/' + args.data_split + '_referenced'
    
class SpacyNER:
    def ner(self,doc):    
        nlp = en_core_web_trf.load()
        doc = nlp(doc)
        return [(X.text, X.label_) for X in doc.ents]
    
    def ner_to_dict(self,ner):
        """
        Expects ner of the form list of tuples 
        """
        ner_dict = {}
        for tup in ner:
            ner_dict[tup[0]] = tup[1]
        return ner_dict
    
    def display(self,ner):
        print(ner)
        print("\n")

def main():
    print("Default ner: spacy")
    
    ner_pickles_op = output_path + "/ner/"
    kg_doc_op = output_path + "/kg/"
    if not os.path.isdir(output_path):
        os.makedirs(output_path) 
        os.makedirs(ner_pickles_op)
        os.makedirs(kg_doc_op)

    file_list = []
    for f in glob.glob(input_path+'/*'):
        file_list.append(f)

    for file in tqdm(file_list):
        with open(file,"r") as f:
            lines = f.read().splitlines()
        
        doc = ""
        for line in lines:
            doc += line + ' '
        
        # extract acts
        act_a = acts.get_act_list(doc)
        act_b = []
        if act_a == []:
            act = []
        else:
            act = []
            for tuple in act_a:
                act.append((tuple['act_name'], 'ACT'))
        
        
        spacy_ner = SpacyNER()
        named_entities = spacy_ner.ner(doc)
        named_entities += act
        named_entities = spacy_ner.ner_to_dict(named_entities)
        
        # Save named entities
        op_pickle_filename = ner_pickles_op + "named_entity_" + file.split('/')[-1].split('.')[0] + ".pickle"
        with open(op_pickle_filename,"wb") as f:
            pickle.dump(named_entities, f)
        op_filename = kg_doc_op + file.split('/')[-1]
        with open(op_filename,"w+") as f:
            f.write(doc)

if __name__ == '__main__':
    main()