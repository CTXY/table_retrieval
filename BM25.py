from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes.retriever import BM25Retriever
from haystack.schema import Document
import jsonlines
import json
import os
from collections import defaultdict
from typing import List, Dict

def convert_file_to_tables(file_path: str) -> List[Document]:
    """ Convert a file to a list of Haystack Document objects representing tables. """
    documents = []
    with jsonlines.open(file_path, 'r') as f:
        for line in f:
            if type(line) == str:
                line = json.loads(line)
            headers = [col['text'] for col in line['columns']]
            # headers = line['columns']
            table_string = f"title: {line['title']}\n" + "| " + " | ".join(headers) + " |\n"
            for row in line['rows'].values():
                row_cells = [cell['text'] for cell in row['cells']]
                table_string += "| " + " | ".join(row_cells) + " |\n"
            documents.append(Document(id=line['id'], content=table_string[:-1], content_type='table'))
    return documents

def load_qrels(file_path: str) -> Dict[str, List[str]]:
    """ Load qrels from a TSV file. """
    qrels = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            qid, _, docid, _ = line.strip().split()
            qrels[qid].append(docid)
    return qrels

def load_queries(file_path: str) -> Dict[str, str]:
    """ Load queries from a TSV file. """
    queries = {}
    with open(file_path, 'r') as f:
        for line in f:
            qid, query = line.strip().split('\t', 1)
            queries[qid] = query
    return queries

def load_questions(file_path: str) -> Dict[str, Dict]:
    """ Load questions from a JSONL file. """
    questions = {}
    with jsonlines.open(file_path) as f:
        for line in f:
            questions[line['id']] = line
    return questions

def calculate_success(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """ Calculate success rate for the retrieved documents. """
    return 1.0 if any(id in retrieved_ids for id in relevant_ids) else 0.0


def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """ Calculate recall rate for the retrieved documents. """
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(relevant_set & retrieved_set) / len(relevant_set)

def run_bm25_retrieval(dataset_name: str):
    """Run BM25 retrieval for a given dataset."""
    base_path = '/home/yangchenyu/table_retrieval/datasets'
    base_result_path = f'/home/yangchenyu/table_retrieval/retrieval_results/{dataset_name}'
    os.makedirs(base_result_path, exist_ok=True)

    file_dir = os.path.join(base_path, dataset_name, 'tables.jsonl')
    documents = convert_file_to_tables(file_dir)

    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(documents)
    retriever = BM25Retriever(document_store)
    
    for split in ['train', 'dev', 'test']:
        questions_path = os.path.join(base_path, dataset_name, split, 'fusion_query.jsonl')
        questions = load_questions(questions_path)
        
        result_path = os.path.join(base_result_path, split)
        os.makedirs(result_path, exist_ok=True)

        results_file_path = os.path.join(result_path, 'results.jsonl')
        existing_queries = set()
        if os.path.exists(results_file_path):
            with jsonlines.open(results_file_path, mode='r') as reader:
                for item in reader:
                    existing_queries.add(item['question_id'])

        queries, qids = [], []
        recalls_dict, successes_dict = defaultdict(list), defaultdict(list)
        with jsonlines.open(results_file_path, 'a') as fout:
            for qid, question in questions.items():
                if qid in existing_queries:
                    continue  # Skip this query if it's already been processed

                query = question['question']
                queries.append(query)
                qids.append(qid)

                if len(queries) == 50:
                    process_queries(queries, qids, questions, retriever, fout, recalls_dict, successes_dict)
                    queries, qids = [], []

            if len(queries) > 0:
                process_queries(queries, qids, questions, retriever, fout, recalls_dict, successes_dict)

        # Calculate metrics after all retrievals
        for k in [1, 5, 10, 20, 50, 100]:
            average_recall = sum(recalls_dict[k]) / len(recalls_dict[k]) if recalls_dict[k] else 0
            average_success = sum(successes_dict[k]) / len(successes_dict[k]) if successes_dict[k] else 0
            print(f"{split.capitalize()} Split - Recall@{k}: {average_recall:.4f}, Success@{k}: {average_success:.4f}")

def process_queries(queries, qids, questions, retriever, fout, recalls_dict, successes_dict):
    batch_results = retriever.retrieve_batch(queries, top_k=100)
    for i in range(len(batch_results)):
        retrieved_ids = [doc.id for doc in batch_results[i]]
        relevant_ids = questions[qids[i]]['table_id_lst']
        for k in [1, 5, 10, 20, 50, 100]:
            recall = calculate_recall(retrieved_ids[:k], relevant_ids)
            recalls_dict[k].append(recall)
            success = calculate_success(retrieved_ids[:k], relevant_ids)
            successes_dict[k].append(success)
        fout.write({'question_id': qids[i], 'results': retrieved_ids})

def main():
    dataset_name = 'fetaqa_dup'
    run_bm25_retrieval(dataset_name)

if __name__ == "__main__":
    main()