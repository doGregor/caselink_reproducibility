# caselink_reproducibility

## Datastructure

To improve the reproducibility of PromtpCase, CaseGNN and CaseLink with new datasets we modified the folder structure of the datasets and the scripts to a more generic setting. Subfolders now do not need specific dataset names or year numbers in their path anymore. Additionally, folders during preprocessing are automatically created and placed in the respective dataset subfolder. In general, all datasets are located in `./datasets/`. The initially required structure in the dataset-specific subfolders looks as follows:

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


## Preparation

Start initially with installing the requirements for this project by running `pip install -r requirements.txt`.

If you want to run the setup based on an OpenAI GPT model you also need to place a file called `openai_api_key.txt` in the root directory of this project which contains a valid OpenAI account key.


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


### 2.5. CaseGNN Model Training and CaseGNN Embedding Generation (Based on CaseGNN)

```
python casegnn/model_training/main.py --in_dim=768 --h_dim=768 --out_dim=768 --dropout=0.1 --num_head=1 --epoch=1000 --lr=5e-6 --wd=5e-5 --batch_size=32 --temp=0.1 --ran_neg_num=1 --hard_neg=True --hard_neg_num=5 --dataset=coliee_2022
```


## 3. CaseLink Graph Construction

```
python caselink/graph_generation/graph_construction.py --dataset coliee_2022 --data_split test --topk_neighbor 5 --charge_threshold 0.9


python caselink/graph_generation/graph_construction.py --dataset coliee_2022 --data_split train --topk_neighbor 5 --charge_threshold 0.9
```


## 4. CaseLink Model Training

```
python caselink/model_training/main.py --in_dim=1536 --h_dim=1536 --out_dim=1536 --dropout=0.2 --epoch=1000 --lr=1e-4 --wd=1e-4 --batch_size=128 --temp=0.1 --hard_neg_num=10 --num_heads=1 --ran_neg_num=1 --layer_num=2 --topk_neighbor=5 --charge_threshold=0.9 --lamb=0.001 --dataset=coliee_2022 --gpu 0
```


# Heterogeneous GNN

Note: To run the setup with heterogeneous graphs and a heterogeneous GNN, all steps up to step 3 have to be completed before.

## 1. Create Heterogeneous Graphs

```
python caselink/graph_generation/heterogeneous_graph_construction.py --dataset coliee_2022 --data_split test --topk_neighbor 5 --charge_threshold 0.9


python caselink/graph_generation/heterogeneous_graph_construction.py --dataset coliee_2022 --data_split train --topk_neighbor 5 --charge_threshold 0.9
```


## 2. Run Heterogeneous CaseLink

```
python caselink/heterogeneous_model_training/main.py --h_dim=1536 --dropout=0.2 --epoch=1000 --lr=0.00001 --wd=0 --batch_size=128 --temp=0.1 --hard_neg_num=5 --num_heads=1 --ran_neg_num=1 --layer_num=2 --topk_neighbor=5 --charge_threshold=0.9 --lamb=0.001 --dataset coliee_2022 --gpu 1
```


# Llama as LLM in Promptcase

Note: Running this requires to first request access to the Llama 3.1 repository on huggingface. More information can be found on the page directly:
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

## Run Summary Generation

```
python promptcase/preprocessing/llama.py --dataset coliee_2022 --data_split test

python promptcase/preprocessing/llama.py --dataset coliee_2022 --data_split train
```

Run the summary generation above instead of `promptcase/preprocessing/openaiAPI.py` in step 2.1.1

From step 2.1.2 on, all commands (except those that are only based on `issue`) can be executed with an additional parameter that indicates the summaries to use. Either `--llm gpt` (GPT 3.5) or `--llm llama` (Llama 8b Instruct).


# PromptCase Baseline

Based on the LLM used to previously generate the case summaries the `--llm` parameter can be left to default `gpt` (ChatGPT 3.5) or set to `llama` (Llama 3.1 8b Instruct).

```
python promptcase/model_training/main.py --dataset coliee_2022 --stage_num 1
```
