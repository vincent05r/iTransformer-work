#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
target=responder_6

python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path p10_train_prep_v2_nd_dropna_lf_row.parquet \
  --model_id JST_v1_t1 \
  --model $model_name \
  --data jst \
  --features MS \
  --target $target \
  --seq_len 1 \
  --label_len 0 \
  --pred_len 1 \
  --c_out 1 \
  --d_model 8 \
  --d_ff 128 \
  --n_heads 2 \
  --e_layers 4 \
  --dropout 0.1 \
  --embed fixed \
  --train_epochs 50 \
  --batch_size 128 \
  --patience 10 \
  --learning_rate 0.001 \
  --lradj typeh_d \
  --des 'JST_V1_t1' \
  --itr 1 > test_s2.txt