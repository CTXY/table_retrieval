import json
import os
from collections import defaultdict

def calculate_metrics(dataset_name: str, split: str):
    base_path = '/home/yangchenyu/table_retrieval'
    result_file = os.path.join(base_path, 'retrieval_results', dataset_name, split, 'results.jsonl')
    query_file = os.path.join(base_path, 'datasets', dataset_name, split, 'fusion_query.jsonl')

    # Load retrieval results
    retrieval_results = {}
    with open(result_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            retrieval_results[data['question_id']] = data['results']

    # Load fusion queries
    fusion_queries = {}
    with open(query_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            fusion_queries[data['id']] = data['table_id_lst']

    # Calculate metrics
    recalls_dict = defaultdict(list)
    successes_dict = defaultdict(list)

    for question_id, retrieved_ids in retrieval_results.items():
        relevant_ids = fusion_queries[question_id]
        for k in [1, 5, 10, 20, 50, 100]:
            recall = calculate_recall(retrieved_ids[:k], relevant_ids)
            recalls_dict[k].append(recall)
            success = calculate_success(retrieved_ids[:k], relevant_ids)
            successes_dict[k].append(success)

    # Print metrics
    for k in [1, 5, 10, 20, 50, 100]:
        average_recall = sum(recalls_dict[k]) / len(recalls_dict[k]) if recalls_dict[k] else 0
        average_success = sum(successes_dict[k]) / len(successes_dict[k]) if successes_dict[k] else 0
        print(f"{split.capitalize()} Split - Recall@{k}: {average_recall:.4f}, Success@{k}: {average_success:.4f}")

def calculate_recall(retrieved_ids, relevant_ids):
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    intersection = retrieved_set.intersection(relevant_set)
    recall = len(intersection) / len(relevant_set)
    return recall

def calculate_success(retrieved_ids, relevant_ids):
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    success = int(bool(retrieved_set.intersection(relevant_set)))
    return success

# Usage example
dataset_name = 'robut-wtq'
split = 'test'
calculate_metrics(dataset_name, split)