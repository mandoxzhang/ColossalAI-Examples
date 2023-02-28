#!/usr/bin/env sh

root_path=$PWD
PY_FILE_PATH="$root_path/run_pretraining.py"

tensorboard_path="$root_path/tensorboard"
log_path="$root_path/exp_log"
ckpt_path="$root_path/ckpt"

colossal_config="$root_path/../configs/colossalai_ddp.py"

mkdir -p $tensorboard_path
mkdir -p $log_path
mkdir -p $ckpt_path

export PYTHONPATH=$PWD

export CUDA_VISIBLE_DEVICES="4,5,6,7"
env OMP_NUM_THREADS=40 colossalai run --hostfile ./hostfile \
                --include GPU009 \
                --nproc_per_node=4 \
                $PY_FILE_PATH \
                --master_addr GPU009 \
                --master_port 55002 \
                --lr 5e-5 \
                --train_micro_batch_size_per_gpu 768 \
                --eval_micro_batch_size_per_gpu 20 \
                --epoch 15 \
                --data_path_prefix /data1/yutian.rong/projects/ColossalAI-Examples/data/train_h5 \
                --eval_data_path_prefix /data1/yutian.rong/projects/ColossalAI-Examples/data/dev_h5 \
                --tokenizer_path /data1/nlp/models/chinese-roberta-wwm-ext-large \
                --bert_config /data1/nlp/models/chinese-roberta-wwm-ext-large/config.json \
                --tensorboard_path $tensorboard_path \
                --log_path $log_path \
                --ckpt_path $ckpt_path \
                --colossal_config $colossal_config \
                --log_interval 50 \
                --mlm bert \
                --wandb \
                --checkpoint_activations \
                --load_pretrain_model /data1/nlp/models/test_model_yutian/pytorch_model.bin \
                