#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
target=responder_6

python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path combined_train_prep_v1_nd.parquet \
  --model_id JST_v1_t1 \
  --model $model_name \
  --data jst \
  --features MS \
  --target $target \
  --seq_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --enc_in 83 \
  --dec_in 83 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1