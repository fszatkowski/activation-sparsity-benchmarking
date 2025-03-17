#!/bin/bash

set -e
source user.env

task=$1
visible_devices=${2:-0}
export CUDA_VISIBLE_DEVICES=${visible_devices}

model_name=meta-llama/Llama-3.2-1B-Instruct
dtype=bfloat16

output_path=results/baseline_benchmarking/${task}/no_template
mkdir -p ${output_path}
python lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=${model_name},trust_remote_code=True,dtype=${dtype} \
    --tasks ${task} \
    --batch_size 1 \
    --output_path ${output_path}

output_path=results/baseline_benchmarking/${task}/chat_template
mkdir -p ${output_path}
python lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=${model_name},trust_remote_code=True,dtype=${dtype} \
    --tasks ${task} \
    --apply_chat_template \
    --batch_size 1 \
    --output_path ${output_path}

