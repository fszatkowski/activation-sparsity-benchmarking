#!/bin/bash

#SBATCH --time=24:00:00   # walltime
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

gradient_accumulation_steps=$1
num_epochs=$2

lr=2e-5
batch_size=2 # Max batch size for Athena jobs
actual_batch_size=$((batch_size*gradient_accumulation_steps))

output_dir=output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-${num_epochs}-bs-${actual_batch_size}-lr-${lr}

python train/main.py \
    --model_name meta-llama/Llama-3.2-1B \
    --output_dir ${output_dir} \
    --run_name llama3-1b-baseline-nepochs-${num_epochs}-bs-${actual_batch_size}-lr-${lr} \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --dataset_name yahma/alpaca-cleaned \
    --dataset_split train \
    --mask_prompt True \
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
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --eval_strategy steps \
    --eval_steps 0.1 \
    --save_strategy steps \
    --save_steps 0.2 \
    --logging_strategy steps \
    --logging_steps 1 \
    --report_to wandb \
    --wandb_tags alpaca
