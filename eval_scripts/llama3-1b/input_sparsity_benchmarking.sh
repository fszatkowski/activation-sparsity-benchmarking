#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --output=logs/%x-%j.out

set -e
source user.env

eval "$(conda shell.bash hook)"
conda activate asb

task=$1
topp=$2
visible_devices=${3:-0}

export CUDA_VISIBLE_DEVICES=${visible_devices}

output_dir=results/llama3-1b/input_sparsification
model_name=meta-llama/Llama-3.2-1B-Instruct
dtype=bfloat16

output_path=${output_dir}/${task}/sparsity_${topp}
mkdir -p ${output_dir}
python lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=${model_name},trust_remote_code=True,dtype=${dtype} \
    --tasks ${task} \
    --batch_size 1 \
    --output_path ${output_path} \
    --sparsity_config sparsity_configs/sparsify_input_llama3-1b.json \
    --sparsification_topp ${topp}
