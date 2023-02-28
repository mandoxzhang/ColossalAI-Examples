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

num_gpus="2"
export CUDA_VISIBLE_DEVICES="1,6"

python -m torch.distributed.launch --nproc_per_node=${num_gpus} --master_port=25251 run_general_distill.py \
                --lr 6e-4 \
                --train_micro_batch_size_per_gpu 64 \
                --eval_micro_batch_size_per_gpu 20 \
                --gradient_accumulation_steps 2 \
                --student_mlm_model_type bert \
                --teacher_mlm_model_type bert \
                --teacher_load_pretrain_model /data1/nlp/models/test_model_yutian/pytorch_model.bin \
                --epoch 15 \
                --max_grad_norm 1.0 \
                --data_path_prefix /data1/yehua.zhang/wudao_h5_new  \
                --eval_data_path_prefix /data1/yehua.zhang/wudao_h5_new_eval \
                --tokenizer_path /data1/nlp/models/chinese-roberta-wwm-ext-large \
                --student_bert_config ./model_config/config.json \
                --teacher_bert_config /data1/nlp/models/chinese-roberta-wwm-ext-large/config.json \
                --tensorboard_path $tensorboard_path \
                --log_path $log_path \
                --ckpt_path $ckpt_path \
                --log_interval 100 \
                --wandb \
                --use_amp \
                --checkpoint_activations \
                --resume_train \
                --student_load_pretrain_model ./ckpt/2023-02-08-13:11:26/epoch-0_shard-7_2023-02-08-13:11:26_pytorch_model.bin \
                --load_optimizer_lr ./ckpt/2023-02-08-13:11:26/epoch-0_shard-7_2023-02-08-13:11:26.op_lrs
                