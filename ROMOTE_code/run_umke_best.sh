#!/bin/bash

DATASET_NAME="UMKE"
BERT_NAME='/data/bert-base-uncased'
VIT_NAME='/data/clip-vit-base-patch32'

CUDA_VISIBLE_DEVICES=1 python run_umke_best.py \
        --dataset_name=${DATASET_NAME} \
        --vit_name=${VIT_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=30 \
        --batch_size=32 \
        --lr=1e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --use_dep \
        --use_box \
        --use_cap \
        --max_seq=128 \
        --save_path="ckpt"