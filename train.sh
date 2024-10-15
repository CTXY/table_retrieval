CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
  --file_dir='/home/yangchenyu/table_retrieval/datasets/generated_data/ours/nq_tables_version_3' \
  --train_filename='train.json' \
  --dev_filename='dev.json' \
  --query_model='/home/yangchenyu/pre-trained-models/bert-base-uncased' \
  --passage_model='/home/yangchenyu/pre-trained-models/bert-base-uncased' \
  --save_dir='/home/yangchenyu/table_retrieval/checkpoints/ours-nq_tables_version_3/DPR' \
  --max_seq_len_query=128 \
  --max_seq_len_passage=512 \
  --training_epochs=50 \
  --batch_size=12 \
  --eval_batch_size=12 \
  --num_positives=1 \
  --num_hard_negatives=7 \
  --evaluate_every=8000 \
  --checkpoint_every=50000 \
  --checkpoint_root_dir='/home/yangchenyu/table_retrieval/temp_checkpoints/ours-nq_tables_version_3/DPR' \
  --dataset_name='ours-nq_tables_version_3'