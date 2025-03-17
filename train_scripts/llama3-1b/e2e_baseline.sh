#!/bin/bash

set -e
source user.env

gpu=$1
batch_size=$2
gradient_accumulation_steps=$3
lr=$4
num_epochs=$5
lora_rank=$6
actual_batch_size=$((batch_size*gradient_accumulation_steps))

export CUDA_VISIBLE_DEVICES=${gpu}

eval_dir=results/baseline_evals

output_dir=output_models/Llama-3.2-1B-Instruct-Alpaca-Finetuned-Intermediate-SE/baseline-lorar-${lora_rank}-nepochs-${num_epochs}-bs-${actual_batch_size}-lr-${lr}
python train/main.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --dataset_name yahma/alpaca-cleaned \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_seq_length 2048 \
    --learning_rate ${lr} \
    --optim adamw_hf \
    --lr_scheduler_type linear \
    --warmup_steps 100 \
    --use_lora True \
    --lora_rank ${lora_rank} \
    --lora_alpha $((2*${lora_rank})) \
    --lora_dropout 0.05 \
    --loss_type l1 \
    --loss_weight 0.0 \
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
    --wandb_tags baseline \
    --run_name llama3-1b-baseline-lorar-${lora_rank}-nepochs-${num_epochs}-bs-${actual_batch_size} \
    --eval_on_start True

for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande gsm8k ifeval mmlu_redux_generative; do
    output_path=${eval_dir}/${task}/llama3-1b-alpaca-epochs-${num_epochs}-bs-${actual_batch_size}-lr-${lr}-lora-rank-${lora_rank}
    mkdir -p ${output_path}
    python lm_eval/__main__.py \
        --model hf \
        --model_args pretrained=${output_dir},trust_remote_code=True,dtype=${dtype} \
        --tasks ${task} \
        --batch_size 1 \
        --output_path ${output_path}
done

