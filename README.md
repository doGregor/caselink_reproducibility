# caselink_reproducibility

## Datastructure

To improve the reproducibility of CaseLine with new datasets we modified the folder structure of the datasets and the scripts to a more generic setting. Subfolders now do not need specific dataset names or year numbers in their path anymore. Additionally, folders during preprocessing are automatically created and placed in the respective dataset subfolder. In general, all datasets are located in `./datasets/`. The structure in the dataset-specific sub-folders looks as follows:

```
    $ ./caselink_reproducibility/
    .
    └── datasets
        ├── coliee_2022
        │   ├── test_files
        │   │   ├── 00001.txt
        │   │   ├── ...
        │   │   └── 99999.txt
        │   ├── train_files
        │   │   ├── 00001.txt
        │   │   ├── ...
        │   │   └── 99999.txt
        │   ├── test_labels.json
        │   └── train_labels.json
        └── your_custom_dataset
            ├── test_files
            │   ├── 00001.txt
            │   ├── ...
            │   └── 99999.txt
            ├── train_files
            │   ├── 00001.txt
            │   ├── ...
            │   └── 99999.txt
            ├── test_labels.json
            └── train_labels.json
```


## 1. Create BM25 Matrices

Run BM25 matrix generation:

`python caselink/bm25/bm25_coliee+lecard.py --ngram_1 1 --ngram_2 1 --dataset coliee_2022 --data_split test`


`python caselink/bm25/bm25_coliee+lecard.py --ngram_1 1 --ngram_2 1 --dataset coliee_2022 --data_split train`



## 2. Generate CaseGNN Embeddings

### 2.1. Legal Information Extraction (Based on PromptCase)

#### 2.1.1. Legal Feature Extraction

```
python promptcase/preprocessing/process.py --dataset coliee_2022 --data_split test

python promptcase/preprocessing/reference.py --dataset coliee_2022 --data_split test

python promptcase/preprocessing/openaiAPI.py --dataset coliee_2022 --data_split test


python promptcase/preprocessing/process.py --dataset coliee_2022 --data_split train

python promptcase/preprocessing/reference.py --dataset coliee_2022 --data_split train

python promptcase/preprocessing/openaiAPI.py --dataset coliee_2022 --data_split train
```


#### 2.1.2. Legal Relation Extraction

`--feature fact` uses summaries and `--feature issue` uses rule based referenced docs.

```
python casegnn/information_extraction/knowledge_graph.py --dataset coliee_2022 --data_split test --feature fact

python casegnn/information_extraction/relation_extractor.py --dataset coliee_2022 --data_split test --feature fact

python casegnn/information_extraction/create_structured_csv.py --dataset coliee_2022 --data_split test --feature fact

python casegnn/information_extraction/knowledge_graph.py --dataset coliee_2022 --data_split test --feature issue

python casegnn/information_extraction/relation_extractor.py --dataset coliee_2022 --data_split test --feature issue

python casegnn/information_extraction/create_structured_csv.py --dataset coliee_2022 --data_split test --feature issue


python casegnn/information_extraction/knowledge_graph.py --dataset coliee_2022 --data_split train --feature fact

python casegnn/information_extraction/relation_extractor.py --dataset coliee_2022 --data_split train --feature fact

python casegnn/information_extraction/create_structured_csv.py --dataset coliee_2022 --data_split train --feature fact

python casegnn/information_extraction/knowledge_graph.py --dataset coliee_2022 --data_split train --feature issue

python casegnn/information_extraction/relation_extractor.py --dataset coliee_2022 --data_split train --feature issue

python casegnn/information_extraction/create_structured_csv.py --dataset coliee_2022 --data_split train --feature issue
```


### 2.2. PromptCase Embedding Generation (Based on PromptCase)

```
python promptcase/embedding/promptcase_embedding_generation.py --dataset coliee_2022 --data_split test


python promptcase/embedding/promptcase_embedding_generation.py --dataset coliee_2022 --data_split train
``` 


### 2.3. TACG Construction (Based on CaseGNN)

```
python casegnn/graph_generation/TACG.py --dataset coliee_2022 --data_split test --feature fact

python casegnn/graph_generation/TACG.py --dataset coliee_2022 --data_split test --feature issue


python casegnn/graph_generation/TACG.py --dataset coliee_2022 --data_split train --feature fact

python casegnn/graph_generation/TACG.py --dataset coliee_2022 --data_split train --feature issue
```

### 2.4. Prepare final Documents before Training (Based on CaseGNN)

```
python casegnn/graph_generation/year_filter.py --dataset coliee_2022


python casegnn/graph_generation/hard_bm25_top50.py --dataset coliee_2022
```


--- TODO FULL AB HIER

### 2.4. CaseGNN Model Training and CaseGNN Embedding Generation (Based on CaseGNN)

```
python casegnn/model_training/main.py --in_dim=768 --h_dim=768 --out_dim=768 --dropout=0.1 --num_head=1 --epoch=600 --lr=5e-5 --wd=5e-5 --batch_size=16 --temp=0.1 --ran_neg_num=1 --hard_neg=True --hard_neg_num=5 --dataset=coliee_2022
```


## 3. CaseLink Graph Construction

```
python caselink/graph_generation/graph_construction.py --data 2022 --dataset test --topk_neighbor 5 --charge_threshold 0.9


python caselink/graph_generation/graph_construction.py --data 2022 --dataset train --topk_neighbor 5 --charge_threshold 0.9
```


## 3. CaseLink Model Training

```
python caselink/model_training/main.py --in_dim=1536 --h_dim=1536 --out_dim=1536 --dropout=0.2 --epoch=100 --lr=1e-4 --wd=1e-4 --batch_size=128 --temp=0.1 --hard_neg_num=10 --num_heads=1 --ran_neg_num=1 --layer_num=2 --topk_neighbor=5 --charge_threshold=0.9 --lamb=0.001 --dataset=coliee_2022
```


# Baselines

## PromptCase

```
python promptcase/model_training/main.py --dataset coliee_2022 --stage_num 1
```
