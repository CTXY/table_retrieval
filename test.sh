CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --model_name='ours-nq_tables_version_3_DPR' \
  --dataset_name='nq_tables' \
  --save_model_dir='/home/yangchenyu/table_retrieval/checkpoints/ours-nq_tables_version_3/DPR' \
  --data_path='/home/yangchenyu/table_retrieval/datasets/nq_tables'

