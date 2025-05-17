#!/bin/bash

set -e
source user.env

# Necessary to download huggingface models and datasets to user dir
export HF_HOME=/net/tscratch/people/plgfszatkowski/huggingface_cache

# Activate conda
eval "$(conda shell.bash hook)"
conda activate asb

eval_dir=results/full_pretraining_alpaca
model_dir_run_name=(
    "meta-llama/Llama-3.2-1B baseline"
    "output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-1-bs-32-lr-2e-5 num_epochs_1_bs_32"
    "output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-1-bs-64-lr-2e-5 num_epochs_1_bs_64"
    "output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-1-bs-128-lr-2e-5 num_epochs_1_bs_128"
    "output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-3-bs-32-lr-2e-5 num_epochs_3_bs_32"
    "output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-3-bs-64-lr-2e-5 num_epochs_3_bs_64"
    "output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-3-bs-128-lr-2e-5 num_epochs_3_bs_128"
    "output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-5-bs-32-lr-2e-5 num_epochs_5_bs_32"
    "output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-5-bs-64-lr-2e-5 num_epochs_5_bs_64"
    "output_models/Alpaca/Llama-3.2-1B/baseline-nepochs-5-bs-128-lr-2e-5 num_epochs_5_bs_128"
)

for params in "${model_dir_run_name[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r model_dir run_name <<< "$params"

    for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande gsm8k ifeval mmlu_redux_generative; do
        # Create a unique job name based on the parameters
        job_name="eval_${run_name}_${task}"
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${job_name} slurm_scripts/alpaca/llama3-1b/evaluate_harness.sh ${model_dir} ${task} ${eval_dir} ${run_name}
    done
done