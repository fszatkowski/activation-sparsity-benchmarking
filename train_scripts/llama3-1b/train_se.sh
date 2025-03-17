#!/bin/bash

set -e
source user.env

batch_size=2
gradient_accumulation_steps=4
num_epochs=3
actual_batch_size=$((batch_size*gradient_accumulation_steps))
lora_rank=16

se_weight=$1
loss_type=$2
gpu=${3:-0}

export CUDA_VISIBLE_DEVICES=${gpu}

output_dir=output_models/Llama-3.2-1B-Instruct-Alpaca-Finetuned-Intermediate-SE/${loss_type}-w${se_weight}-lorar-${lora_rank}-nepochs-${num_epochs}-bs-${actual_batch_size}
python train/main.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name yahma/alpaca-cleaned \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_seq_length 2048 \
    --learning_rate 5e-5 \
    --optim adamw_hf \
    --lr_scheduler_type linear \
    --warmup_steps 100 \
    --use_lora True \
    --lora_rank ${lora_rank} \
    --lora_alpha $((2*${lora_rank})) \
    --lora_dropout 0.05 \
    --loss_type ${loss_type} \
    --loss_weight ${se_weight} \
    --modules_to_sparsify mlp.down_proj \
    --sparsification_modes input \
    --modules_to_monitor mlp.down_proj mlp \
    --monitor_modes input input \
    --monitor_top_p 0.50 0.75 0.90 0.95 0.99 0.999 \
    --eval_strategy steps \
    --eval_steps 0.1 \
    --logging_strategy steps \
    --logging_steps 1 \
    --report_to wandb \
    --run_name llama3-1b-${loss_type}-w${se_weight}-lorar-${lora_rank}-nepochs-${num_epochs}-bs-${actual_batch_size} \
    --eval_on_start True
