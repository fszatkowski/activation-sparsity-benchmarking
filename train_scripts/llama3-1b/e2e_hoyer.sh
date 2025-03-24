#!/bin/bash

set -e
source user.env

gpu=$1
se_weight=$2

batch_size=2
gradient_accumulation_steps=4
lr=5e-5
num_epochs=3
lora_rank=16
loss_type=hoyer
actual_batch_size=$((batch_size*gradient_accumulation_steps))

export CUDA_VISIBLE_DEVICES=${gpu}

eval_dir=results/full_model_evals

output_dir=output_models/Llama-3.2-1B-Instruct-Alpaca-Finetuned-Intermediate-SE/${loss_type}-lorar-${lora_rank}-nepochs-${num_epochs}-bs-${actual_batch_size}-lr-${lr}-se-${se_weight}
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
    --wandb_tags hoyer \
    --run_name llama3-1b-${loss_type}-lorar-${lora_rank}-nepochs-${num_epochs}-bs-${actual_batch_size}-se-${se_weight} \
    --eval_on_start True

dtype=bfloat16
for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande gsm8k ifeval mmlu_redux_generative; do
    output_path=${eval_dir}/${task}/llama3-1b-alpaca-${loss_type}-epochs-${num_epochs}-bs-${actual_batch_size}-lr-${lr}-lora-rank-${lora_rank}-se-${se_weight}
    mkdir -p ${output_path}
    python lm_eval/__main__.py \
        --model hf \
        --model_args pretrained=${output_dir},trust_remote_code=True,dtype=${dtype} \
        --tasks ${task} \
        --batch_size 1 \
        --output_path ${output_path}
done

