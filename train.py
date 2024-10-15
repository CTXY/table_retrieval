import logging
import argparse 
from src.retriever import DensePassageRetriever, SiameseRetriever
from src.UTP import UniversalTableRetriever
from haystack.document_stores import InMemoryDocumentStore
import torch
import os

def main():
    parser = argparse.ArgumentParser(description="Tuple Learning for retrieval")
    
    # model setting
    parser.add_argument('--file_dir', required=True, type=str, help='file that contains necessary data')
    parser.add_argument('--train_filename', required=True, type=str, help='name of training file')
    parser.add_argument('--dev_filename', required=True, type=str, help='name of training file')
    parser.add_argument('--query_model', required=False, type=str, help='query model')
    parser.add_argument('--passage_model', required=False, type=str, help='passage model')
    parser.add_argument('--general_model', required=False, type=str, help='same model for query and passage')
    parser.add_argument('--max_seq_len', default=512, type=int, required=False, help='maximun length of query and passage')
    parser.add_argument('--max_seq_len_query', default=256, type=int, required=False, help='maximun length of query')
    parser.add_argument('--max_seq_len_passage', default=256, type=int, required=False, help='maximun length of passage')

    # training setting
    parser.add_argument('--training_epochs', default=10, type=int, required=False, help='')
    parser.add_argument('--evaluate_every', default=5, type=int, required=False, help='')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='')
    parser.add_argument('--eval_batch_size', default=32, type=int, required=False, help='')
    parser.add_argument('--num_positives', default=1, type=int, required=False, help='')
    parser.add_argument('--num_hard_negatives', default=3, type=int, required=False, help='')
    parser.add_argument('--learning_rate', default=1e-5, type=float, required=False, help='')
    parser.add_argument('--checkpoint_every', default=1000, type=int, required=False, help='')
    parser.add_argument('--checkpoint_root_dir', default='', type=str, required=False, help='')
    parser.add_argument('--dataset_name', default='OTT_QA', type=str, required=False, help='')
    parser.add_argument('--query_structure', default='global', type=str, required=False, help='')
    parser.add_argument('--table_structure', default='global', type=str, required=False, help='')
    parser.add_argument('--model_architecture', default='default', type=str, required=False, help='')
    parser.add_argument('--linearization', default='default', type=str, required=False, help='')
    parser.add_argument('--linearization_direction', default='row', type=str, required=False, help='')
    
    # output setting
    parser.add_argument('--save_dir', required=True, type=str, help='')

    args = parser.parse_args()

    # 打印parse的所有config参数
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")


    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Find {gpu_count} available GPU.")
    
    if args.model_architecture == 'default':
        retriever = DensePassageRetriever(
            document_store=InMemoryDocumentStore(),
            query_embedding_model=args.query_model,
            passage_embedding_model=args.passage_model,
            max_seq_len_query=args.max_seq_len_query,
            max_seq_len_passage=args.max_seq_len_passage,
            query_structure=args.query_structure,
            table_structure=args.table_structure,
            linearization=args.linearization,
            linearization_direction=args.linearization_direction,
        )
    elif args.model_architecture == 'UTP':
        retriever = UniversalTableRetriever(
            document_store=InMemoryDocumentStore(),
            embedding_model=args.general_model,
            max_seq_len=args.max_seq_len,
        )
    elif args.model_architecture == 'Siamese':
        retriever = SiameseRetriever(
            document_store=InMemoryDocumentStore(),
            embedding_model=args.passage_model,
            max_seq_len=args.max_seq_len_passage,
            structure=args.table_structure,
            linearization=args.linearization,
            linearization_direction=args.linearization_direction,
        )

    retriever.train(
        data_dir=args.file_dir,
        learning_rate=args.learning_rate,
        train_filename=args.train_filename,
        dev_filename=args.dev_filename,
        test_filename=args.dev_filename,
        n_epochs=args.training_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_acc_steps=1,
        save_dir=args.save_dir,
        evaluate_every=args.evaluate_every,
        checkpoint_every=args.checkpoint_every,
        checkpoints_to_keep=10,
        checkpoint_root_dir=args.checkpoint_root_dir,
        embed_title=False,
        num_positives=args.num_positives,
        num_hard_negatives=args.num_hard_negatives,
    )

if __name__ == "__main__":
    main()