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

### 2.1. Legal Information Extraction (using PromptCase)

#### 2.1.1 Legal Feature Extraction

```
python promptcase/preprocessing/process.py --dataset coliee_2022 --data_split test

python promptcase/preprocessing/reference.py --dataset coliee_2022 --data_split test

python promptcase/preprocessing/openaiAPI.py --dataset coliee_2022 --data_split test

python promptcase/preprocessing/process.py --dataset coliee_2022 --data_split train

python promptcase/preprocessing/reference.py --dataset coliee_2022 --data_split train

python promptcase/preprocessing/openaiAPI.py --dataset coliee_2022 --data_split train
```

#### 2.1.2 Legal Relation Extraction

```
python casegnn/information_extraction/knowledge_graph.py --dataset coliee_2022 --data_split test --feature issue

python Information_extraction/relation_extractor.py --data 2022 --dataset train --feature fact

python Information_extraction/create_structured_csv.py --data 2022 --dataset train --feature fact

python Information_extraction/knowledge_graph.py --data 2022 --dataset train --feature issue

python Information_extraction/relation_extractor.py --data 2022 --dataset train --feature issue

python Information_extraction/create_structured_csv.py --data 2022 --dataset train --feature issue

python Information_extraction/knowledge_graph.py --data 2022 --dataset test --feature fact

python Information_extraction/relation_extractor.py --data 2022 --dataset test --feature fact

python Information_extraction/create_structured_csv.py --data 2022 --dataset test --feature fact

python Information_extraction/knowledge_graph.py --data 2022 --dataset test --feature issue

python Information_extraction/relation_extractor.py --data 2022 --dataset test --feature issue

python Information_extraction/create_structured_csv.py --data 2022 --dataset test --feature issue
```
