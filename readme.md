# Project Overview
This repository contains the implementation of the paper "**Dense Table Retrieval from Data Lakes for Natural Language Questions: A Design Space Exploration and Evaluation**".

The project explores dense retrieval techniques for natural language (NL) questions over tables. It systematically evaluates different model architectures, training strategies, table structure encoding, and linearization methods to optimize table retrieval.

## Key features:
 - **Design Space Exploration**:  We  explore different model architectures, encoding methods, training strategies, and table linearization techniques. Our code allows you to experiment with different combinations within this design space.
 - **Benchmark Datasets**: we collect and curate six datasets for table retrieval. You can train and test models across various datasets, with flexibility to choose the dataset that suits your needs.
- **Training Data Generation**:  We provides an algorithm to automatically generate synthetic training data. You can use this algorithm to generate training data on the provided datasets or apply it to your own data.

This project is built on the [Haystack](https://github.com/deepset-ai/haystack), an open source framework for building production-ready LLM applications, retrieval-augmented generative pipelines and state-of-the-art search systems that work intelligently over large document collections. 

# Environment Setup

To set up the environment, you can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

# Datasets and Pre-Trained Models

[Download Data and Pre-Trained Models](https://osf.io/k8d3q/?view_only=cc4610d536404db1912fb2a7fd79aed7)

## Dataset 
You can download the datasets in our benchmark, which include the following six datasets:

| **Dataset**        | **# of Questions (train/dev/test)** | **# of Tables** |
|--------------------|-------------------------------------|-----------------|
| **FeTaQA**         | 7,326 / 1,001 / 2,003               | 10,330          |
| **NQ-TABLES**      | 8,442 / 943 / 958                   | 127,629         |
| **WikiSQL**        | 14,888 / 1,654 / 3,722              | 26,531          |
| **WTQ**            | 4,223 / 628 / 1,020                 | 2,108           |
| **OTT-QA**         | 6,000 / 600 / 876                   | 419,183         |
| **MMQA**           | 3,932 / 491 / 492                   | 10,041          |

Alternatively, you can create your own dataset. You will need to create a `train.json` file and `dev.json` with the following format:

```json
{
    "dataset": "str",
    "question": "str",
    "answers": ["list of str"],
    "positive_ctxs": [
        {
            "title": "str",
            "text": "str",
            "score": "int",
            "title_score": "int",
            "passage_id": "str"
        }
    ],
    "negative_ctxs": [
        {
            "title": "str",
            "text": "str",
            "score": "int",
            "title_score": "int",
            "passage_id": "str"
        }
    ],
    "hard_negative_ctxs": [
        {
            "title": "str",
            "text": "str",
            "score": "int",
            "title_score": "int",
            "passage_id": "str"
        }
    ]
}
```
 - positive_ctxs: Relevant tables to the query. In some datasets, tables might have more than one positive context, in which case you can set the num_positives parameter higher than the default 1.

**Table Data Format** 
To store table data, you need to create a `tables.jsonl` file where each table is represented with the following fields:
```json
{
    "id": "str",  
    "title": "str",  
    "columns": [  
        {"text": "str"}
    ],
    "cells": [  
        {
            "text": "str",  
            "row_idx": "int", 
            "col_idx": "int" 
        },
    ],
    "rows": { 
        "row_index": {
            "cells": [ 
                {"text": "str"},  
            ]
        },
    }
}

```
**Test Data Format**
Your test data should be stored in a `test.jsonl` file with the following structure:
```json
{
    "id": "str", 
    "question": "str", 
    "table_id_lst": ["str"],
    "answers": ["list of str"], 
    "ctxs": [
        {
            "title": "str", 
            "text": "str" 
        },
    ]
}
```
## Pre-Trained Models

We sampled 1.5 million text-table pairs from the [DTR dataset](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md), filtering out tables with excessive missing values. Using TAPAS-base as the backbone model, we re-trained both UTP and DTR models.

You can download the pre-trained UTP and DTR checkpoints trained by us. Alternatively, you can run the original DTR models and code from its [official repository](https://github.com/google-research/tapas/).

- **DTR**: [Open domain question answering over tables via dense retrieval](https://arxiv.org/abs/2103.12011), Herzig J, Müller T, Krichene S, et al., 2021.
- **UTP**: [Bridge the gap between language models and tabular understanding](https://arxiv.org/abs/2302.09302), Chen N, Shou L, Gong M, et al., 2023.

# Model Training
This section outlines how to train the model for dense table retrieval using the provided training scripts. You can explore the design space by selecting different model architectures, encoding strategies, and linearization methods.
###  Model Architecture
You can use the following commands to train the model. Here is an explanation of the main parameters:

- `file_dir`: Specifies the directory where your training data is stored.
- `query_model` and `passage_model`: These represent the base models for the two encoders. We support both **BERT-base-uncased** and **TAPAS-base**.
- `save_dir`: The directory where the trained model will be saved.
- `num_positives`: must be less than or equal to the number of positive_ctxs available for queries in your dataset. 
- `num_hard_negatives`: must be less than or equal to the number of hard_negative_ctxs available for queries in your dataset. 

**Siamese Dual Encoder**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --file_dir='./datasets/fetaqa' \
  --train_filename='train.json' \
  --dev_filename='dev.json' \
  --passage_model='bert-base-uncased' \
  --save_dir='./checkpoints/fetaqa/Siamese_BERT' \
  --max_seq_len_passage=512 \
  --training_epochs=50 \
  --batch_size=12 \
  --eval_batch_size=12 \
  --num_positives=1 \
  --num_hard_negatives=7 \
  --checkpoint_root_dir='./temp_checkpoints/ott-qa/pretrained_UTP_ott-qa' \
  --model_architecture='Siamese'
```

**Asymmetric Dual Encoder**

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --file_dir='./datasets/fetaqa' \
  --train_filename='train.json' \
  --dev_filename='dev.json' \
  --query_model='bert-base-uncased' \
  --passage_model='bert-base-uncased' \
  --save_dir='./checkpoints/fetaqa/DPR' \
  --max_seq_len_query=128 \
  --max_seq_len_passage=512 \
  --training_epochs=50 \
  --batch_size=12 \
  --eval_batch_size=12 \
  --num_positives=1 \
  --num_hard_negatives=7 \
  --checkpoint_root_dir='./temp_checkpoints/fetaqa/DPR'
```

###   Pre-Training
To fine-tune based on a pre-trained **UTP** or DTR model, you can download the corresponding checkpoints:

- **UTP**: Use the SDE training code and replace `passage_model` with the path to the UTP checkpoint.
- **DTR**: Use the ADE training code and replace `query_model` and `passage_model` with the respective DTR checkpoint paths.
### Encoding Strategies
You can specify different encoding strategies for the `query_structure` and `table_structure` parameters:

- **bias**:  Attention bias.
- **rowcol**: Attention mask.
- **auxemb**: Positional embeddings.

Example:

```bash
--query_structure='bias' \
--table_structure='bias'
```
### Linearization

You can linearize tables either by row or column and choose among different linearization methods:

 **Linearization Direction**:
  - `row`: Linearize the table by row.
  - `column`: Linearize the table by column.

**Linearization Methods**:
  - `direct`: Directly linearize the table without extra formatting.
  - `default`: Use simple delimiters (e.g., commas or spaces).
  - `separator`: Insert special tokens to separate elements (e.g., [SEP] tokens).
  - `template`: Use a textual template to format the table into natural language.

Example:

```bash
--linearization_direction='row' \
--linearization='direct'
```
# Table Retrieval

Once you have a trained model, you can use the following code to encode embeddings and perform retrieval:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --model_name='DPR' \
  --dataset_name='nq_tables' \
  --save_model_dir='./checkpoints/fetaqa/DPR' \
  --data_path='/home/yangchenyu/table_retrieval/datasets/fetaqa'
```

- `data_path`: Must contain your `test.jsonl` file and `tables.jsonl` file.
- `save_model_dir`: Path where your trained model is stored.


# Training Data Generation

To generate training data on a new table collection, please refer to the instructions provided in the [question_generator/readme.md](./question_generator/readme.md) and follow the steps outlined there.