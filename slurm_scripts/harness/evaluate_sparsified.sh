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

model_dir=$1
task=$2
eval_dir=$3
run_name=$4
sparsification_config=$5
sparsification_rule=$6
sparsification_th=$7

dtype=float32
output_path=${eval_dir}/${task}/${run_name}
mkdir -p ${output_path}
python lm_eval/__main__.py \
    --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,dtype=${dtype},add_bos_token=True \
    --tasks ${task} \
    --batch_size auto \
    --output_path ${output_path} \
    --sparsification_config ${sparsification_config} \
    --sparsification_rule ${sparsification_rule} \
    --sparsification_th_val ${sparsification_th}



