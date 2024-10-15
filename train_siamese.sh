CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
  --file_dir='/home/yangchenyu/table_retrieval/datasets/ott-qa' \
  --train_filename='train.json' \
  --dev_filename='dev.json' \
  --passage_model='/home/yangchenyu/table_retrieval/checkpoints/pretrained_UTP' \
  --save_dir='/home/yangchenyu/table_retrieval/checkpoints/ott-qa/pretrained_UTP_ott-qa' \
  --max_seq_len_passage=512 \
  --training_epochs=50 \
  --batch_size=12 \
  --eval_batch_size=12 \
  --num_positives=1 \
  --num_hard_negatives=7 \
  --evaluate_every=8000 \
  --checkpoint_every=50000 \
  --checkpoint_root_dir='/home/yangchenyu/table_retrieval/temp_checkpoints/ott-qa/pretrained_UTP_ott-qa' \
  --dataset_name='ott-qa' \
  --model_architecture='Siamese'