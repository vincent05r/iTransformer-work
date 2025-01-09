#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=nlinear
target=responder_6

python -u run.py \
  --is_training 1 \
  --root_path ./data/ \
  --data_path p5_p9_train_prep_v2_nd_dropna_lf_row.parquet \
  --model_id JST_v1_t1 \
  --model nlinear \
  --data jst \
  --features MS \
  --target responder_6 \
  --seq_len 1 \
  --label_len 0 \
  --pred_len 1 \
  --enc_in 66 \
  --c_out 1 \
  --individual \
  --train_epochs 50 \
  --batch_size 128 \
  --patience 10 \
  --learning_rate 0.001 \
  --lradj typeh_d \
  --des 'JST_V1_t1' \
  --itr 1 > test_v2_half_l.txt