#!/bin/bash

set -e
source user.env

# Necessary to download huggingface models and datasets to user dir
export HF_HOME=/net/tscratch/people/plgfszatkowski/huggingface_cache

# Activate conda
eval "$(conda shell.bash hook)"
conda activate asb

eval_dir=results/full_pretraining_alpaca_ablations
model_dir_run_name=(
    "meta-llama/Llama-3.2-1B baseline"
    # "output_models/Alpaca/Llama-3.2-1B-ablations/baseline-nepochs-1-bs-128-lr-2e-5-mask_prompt-True num_epochs_1_bs_128_mask_prompt_True"
    # "output_models/Alpaca/Llama-3.2-1B-ablations/baseline-nepochs-1-bs-128-lr-2e-5-mask_prompt-False num_epochs_1_bs_128_mask_prompt_False"
    "output_models/Alpaca/Llama-3.2-1B-ablations/baseline-nepochs-3-bs-128-lr-2e-5-mask_prompt-True num_epochs_3_bs_128_mask_prompt_True"
    "output_models/Alpaca/Llama-3.2-1B-ablations/baseline-nepochs-3-bs-128-lr-2e-5-mask_prompt-False num_epochs_3_bs_128_mask_prompt_False"
)

for params in "${model_dir_run_name[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r model_dir run_name <<< "$params"

    for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande gsm8k ifeval mmlu_redux_generative; do
        # Create a unique job name based on the parameters
        job_name="eval_${run_name}_${task}"
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${job_name} slurm_scripts/alpaca/llama3-1b-ablations/evaluate_harness.sh ${model_dir} ${task} ${eval_dir} ${run_name}
    done
done