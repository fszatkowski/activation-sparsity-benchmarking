#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --cpus-per-task=16   # number of processor cores (i.e. tasks)
#SBATCH --mem=32G   # memory (see also --mem-per-cpu)
#SBATCH --gpus=1

set -e
source user.env

# Necessary to download huggingface models and datasets to user dir
export HF_HOME=/net/tscratch/people/plgfszatkowski/huggingface_cache

# Activate conda
eval "$(conda shell.bash hook)"
conda activate asb

lr=$1
num_samples=$2
gradient_accumulation_steps=$3
relufication_mode=$4
output_dir=$5

num_epochs=1
batch_size=2 # Max batch size for Athena jobs

actual_batch_size=$((batch_size*gradient_accumulation_steps))


python train/main.py \
    --model_name meta-llama/Llama-3.2-1B \
    --output_dir ${output_dir} \
    --run_name llama3-1b-fineweb-s${num_samples}-gate-relufication-${relufication_mode}-nepochs-${num_epochs}-bs-${actual_batch_size}-lr-${lr} \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --dataset_name HuggingFaceFW/fineweb \
    --dataset_split train[:${num_samples}] \
    --test_size 0.01 \
    --mask_prompt False \
    --max_seq_length 2048 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --dataloader_pin_memory True \
    --eval_on_start True \
    --optim adamw_torch \
    --learning_rate ${lr} \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --loss_type none \
    --loss_weight 0.0 \
    --modules_to_sparsify [] \
    --sparsification_modes [] \
    --modules_to_monitor mlp mlp.down_proj mlp.act_fn \
    --relufication True \
    --relufication_mode ${relufication_mode} \
    --relufication_target_modules layers.7.mlp.act_fn layers.8.mlp.act_fn \
    --monitor_modes input input output \
    --eval_strategy steps \
    --eval_steps 0.05 \
    --save_strategy no \
    --save_steps 0.2 \
    --logging_strategy steps \
    --logging_steps 1 \
    --report_to wandb \
    --wandb_tags fineweb-s${num_samples}-gate-relufication-${relufication_mode}
