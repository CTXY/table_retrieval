import os
import argparse
from collections import defaultdict

from src.document_stores.faiss import FAISSDocumentStore
from haystack.document_stores import InMemoryDocumentStore
from src.retriever import DensePassageRetriever, SiameseRetriever
from src.UTP import UniversalTableRetriever
from utils import convert_file_to_table
import jsonlines

import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.modeling_utils')

base_path = os.path.dirname(os.path.abspath(__file__))

def calculate_recall(topk_pids, qrels, K):
    recall_sum = 0.0
    num_queries = len(topk_pids)

    for qid, retrieved_docs in topk_pids.items():
        retrieved_docs = set(retrieved_docs[:K])
        relevant_docs = set(qrels[qid])

        intersection = relevant_docs.intersection(retrieved_docs)
        recall = len(intersection) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
        recall_sum += recall
 
    # 计算平均Recall Rate
    recall_rate = recall_sum / num_queries
    recall_rate = round(recall_rate, 3)
    print("Recall@{} =".format(K), recall_rate)
    return recall_rate
    

def calculate_success(topk_pids, qrels, K):
    success_at_k = []

    for qid, retrieved_docs in topk_pids.items():
        topK_docs = set(retrieved_docs[:K])
        relevant_docs = set(qrels[qid])

        if relevant_docs.intersection(topK_docs):
            success_at_k.append(1)

    success_at_k_avg = sum(success_at_k) / len(qrels)
    success_at_k_avg = round(success_at_k_avg, 3)
    
    print("Success@{} =".format(K), success_at_k_avg)
    return success_at_k_avg
 

def evaluate_on_dev(args):
    best_model_file = ''
    best_recall = 0.0
    
    if os.path.exists(args.save_model_dir) and os.path.isdir(args.save_model_dir):
        subfolders = [f for f in os.listdir(args.save_model_dir) if os.path.isdir(os.path.join(args.save_model_dir, f))]
        print("Subfolders:", subfolders)
        docs = convert_file_to_table(file_path=os.path.join(args.data_path, 'tables.jsonl'))
        
        dev_qrels = defaultdict(list)
        with jsonlines.open(os.path.join(args.data_path, 'test/fusion_query.jsonl'), 'r') as f:
            for line in f:
                dev_qrels[line['id']] = line['table_id_lst']
        
        for folder in subfolders:
            if folder.startswith('epoch_') and '_step_' in folder:
                print("-----------------")
                print("Evaluating model:", folder)
                args.model_name = f"{args.model_name}_{folder}"
                print(args.model_name)
                eval_step(args, folder, docs, dev_qrels, 'test')
                


def eval_step(args, model_dir, docs, qrels, fold_name='dev'):
    if args.model_architecture == 'default':
        retriever = DensePassageRetriever.load(
            document_store=InMemoryDocumentStore(), 
            load_dir=os.path.join(args.save_model_dir, model_dir), 
            batch_size=64,
            max_seq_len_query=args.max_seq_len_query, 
            max_seq_len_passage=args.max_seq_len_passage,
            query_structure=args.query_structure,
            table_structure=args.table_structure,
            linearization=args.linearization,
            linearization_direction=args.linearization_direction
        )
    elif args.model_architecture == 'UTP':
        retriever = UniversalTableRetriever.load(
            document_store=InMemoryDocumentStore(),
            load_dir=os.path.join(args.save_model_dir, model_dir), 
            max_seq_len=args.max_seq_len,
        )
    elif args.model_architecture == 'Siamese':
        retriever = SiameseRetriever.load(
            document_store=InMemoryDocumentStore(),
            load_dir=os.path.join(args.save_model_dir, model_dir), 
            max_seq_len=args.max_seq_len,
            structure=args.table_structure,
            linearization=args.linearization,
            linearization_direction=args.linearization_direction
        )


    index_name = f"{args.dataset_name}_{args.model_name}"
    index_path = os.path.join(base_path, f'index/{index_name}.faiss')
    
    print(f"---------------Index Name: {index_name}-----------------")

    if not os.path.exists(index_path):
        document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", index=index_name)

        document_store.write_documents(docs)

        document_store.update_embeddings(retriever)
        document_store.save(index_path=index_path)
    else:
        document_store = FAISSDocumentStore(faiss_index_path=index_path)

    rank_result = {}

    result_file = os.path.join(base_path, f'retrieval_results/{args.dataset_name}/{args.model_name}_{fold_name}_top100_res.jsonl')
    query_file = os.path.join(base_path, f'datasets/{args.dataset_name}/{fold_name}/fusion_query.jsonl')
    # query_file = '/home/yangchenyu/table_retrieval/datasets/robut-wikisql/test/fusion_query.jsonl'

    batch_size = 64  # 指定每次处理的查询数量

    existing_query_ids = set()
    try:
        with jsonlines.open(result_file) as f_reader:
            for line in f_reader:
                if 'question_id' in line:
                    existing_query_ids.add(line['question_id'])
    except FileNotFoundError:
        # 如果result_file不存在，则继续正常流程
        pass

    # 现在处理新的查询，跳过已存在的查询
    with jsonlines.open(result_file, mode='a') as f_writer:  # 使用追加模式'a'
        queries = []
        query_ids = []

        with jsonlines.open(query_file) as f:
            for line in f:
                qid, query = line['id'], line['question']
                if qid not in existing_query_ids:  # 只处理不在existing_query_ids的查询
                    queries.append(query)
                    query_ids.append(qid)

                    if len(queries) == batch_size:
                        top_documents_batch = retriever.retrieve_batch(queries, top_k=100, document_store=document_store, batch_size=batch_size)

                        for qid, top_documents in zip(query_ids, top_documents_batch):
                            document_ids = [document.id for document in top_documents]
                            f_writer.write({'question_id': qid, 'results': document_ids})

                            rank_result[qid] = document_ids

                        queries = []
                        query_ids = []

        # 处理剩余的查询（如果有）
        if queries:
            top_documents_batch = retriever.retrieve_batch(queries, top_k=100, document_store=document_store, batch_size=len(queries))

            for qid, top_documents in zip(query_ids, top_documents_batch):
                document_ids = [document.id for document in top_documents]
                f_writer.write({'question_id': qid, 'results': document_ids})

                rank_result[qid] = document_ids

    for K in [1, 5, 10, 20, 50, 100]:
        calculate_recall(rank_result, qrels, K)
        calculate_success(rank_result, qrels, K)


def main():
    parser = argparse.ArgumentParser(description="Table Retrieval")

    parser.add_argument('--model_name', required=True, default='DPR', type=str, help='name of the model')
    parser.add_argument('--dataset_name', required=True, default='wikituples', type=str, help='name of the dataset')
    parser.add_argument('--save_model_dir', required=True, type=str, help='directory that saves the model')
    parser.add_argument('--data_path', required=True, type=str, help='path of data')
    parser.add_argument('--max_seq_len_query', required=False, default=128, type=int, help='')
    parser.add_argument('--max_seq_len_passage', required=False, default=512, type=int, help='')
    parser.add_argument('--max_seq_len', required=False, default=512, type=int, help='')
    parser.add_argument('--query_structure', required=False, default='global', type=str, help='')
    parser.add_argument('--table_structure', required=False, default='global', type=str, help='')
    parser.add_argument('--same_structure', action='store_true', help='Enable if structures should be the same')
    parser.add_argument('--linearization', default='default', type=str, required=False, help='')
    parser.add_argument('--linearization_direction', default='row', type=str, required=False, help='')
    parser.add_argument('--model_architecture', default='default', type=str, required=False, help='')

    args = parser.parse_args()

    # best_model_file = evaluate_on_dev(args)

    test_qrels = defaultdict(list)
    
    with jsonlines.open(os.path.join(base_path, f'datasets/{args.dataset_name}/test/fusion_query.jsonl'), 'r') as f:
        # with jsonlines.open('/home/yangchenyu/table_retrieval/datasets/robut-wikisql/test/fusion_query.jsonl', 'r') as f:
        for line in f:
            test_qrels[line['id']] = line['table_id_lst']

    docs = convert_file_to_table(file_path=os.path.join(args.data_path, 'tables.jsonl'))
    eval_step(args, args.save_model_dir, docs, test_qrels, 'test')


    # if best_model_file:
    #     test_qrels = defaultdict(list)

    #     with jsonlines.open(os.path.join(base_path, f'datasets/{args.dataset_name}/test/fusion_query.jsonl'), 'r') as f:
    #         for line in f:
    #             test_qrels[line['id']] = line['table_id_lst']

    #     docs = convert_file_to_table(file_path=os.path.join(args.data_path, 'tables.jsonl'))
    #     eval_step(args, best_model_file, docs, test_qrels, 'test')

if __name__ == "__main__":
    main()