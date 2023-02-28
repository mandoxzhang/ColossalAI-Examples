#!/usr/bin/env sh

root_path=$PWD
PY_FILE_PATH="$root_path/run_pretraining.py"

tensorboard_path="$root_path/tensorboard"
log_path="$root_path/exp_log"
ckpt_path="$root_path/ckpt"

mkdir -p $tensorboard_path
mkdir -p $log_path
mkdir -p $ckpt_path

export PYTHONPATH=$PWD

num_gpus="1"
export CUDA_VISIBLE_DEVICES="7"


#Distillation phase1
python -m torch.distributed.launch --nproc_per_node=${num_gpus} --master_port=25251 run_pretrain_ddp.py \
                --lr 6e-4 \
                --train_micro_batch_size_per_gpu 5 \
                --eval_micro_batch_size_per_gpu 20 \
                --gradient_accumulation_steps 1 \
                --mlm_model_type bert \
                --load_pretrain_model /data1/nlp/models/test_model_yutian/pytorch_model.bin \
                --epoch 15 \
                --max_grad_norm 1.0 \
                --data_path_prefix /data1/yehua.zhang/wudao_h5_new  \
                --eval_data_path_prefix /data1/yehua.zhang/wudao_h5_new_eval \
                --tokenizer_path /data1/nlp/models/chinese-roberta-wwm-ext-large \
                --bert_config /data1/nlp/models/chinese-roberta-wwm-ext-large/config.json \
                --tensorboard_path $tensorboard_path \
                --log_path $log_path \
                --ckpt_path $ckpt_path \
                --log_interval 100 \
                --checkpoint_activations \
                --wandb
                # --student_load_pretrain_model /data1/nlp/models/bert-small/pytorch_model.bin \
