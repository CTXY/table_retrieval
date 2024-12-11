# Project Overview
This repository contains the implementation of the paper "**Dense Table Retrieval from Data Lakes for Natural Language Questions: A Design Space Exploration and Evaluation**".

The project explores dense retrieval techniques for natural language (NL) questions over tables. It systematically evaluates different model architectures, training strategies, table structure encoding, and linearization methods to optimize table retrieval.

This project is built on the [Haystack](https://github.com/deepset-ai/haystack), an open source framework for building production-ready LLM applications, retrieval-augmented generative pipelines and state-of-the-art search systems that work intelligently over large document collections. 

# Environment Setup

To set up the environment, you can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

# Datasets and Pre-Trained Models

## Dataset 
You can create your own dataset. You will need to create a `train.json` file and `dev.json` with the following format:

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

# Model Training
This section outlines how to train the model for dense table retrieval using the provided training scripts. You can explore the design space by selecting different model architectures, encoding strategies, and linearization methods.
###  Model Architecture
You can use the following commands to train the model. Here is an explanation of the main parameters:

- `file_dir`: Specifies the directory where your training data is stored.
- `query_model` and `passage_model`: These represent the base models for the two encoders. We support both **BERT-base-uncased** and **TAPAS-base**.
- `save_dir`: The directory where the trained model will be saved.
- `num_positives`: must be less than or equal to the number of positive_ctxs available for queries in your dataset. 
- `num_hard_negatives`: must be less than or equal to the number of hard_negative_ctxs available for queries in your dataset. 


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
