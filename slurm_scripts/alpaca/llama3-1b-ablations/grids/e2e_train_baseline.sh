#!/bin/bash

# Create a list of tuples for batch sizes and num epochs that we will iterate over
# Tuple structure: "batch_size num_epochs"
hyperparams=(
    "64 1 False"
    "64 1 True"
    "64 3 False"
    "64 3 True"
)

# Loop over hyperparams and run the training scripts through slurm
for params in "${hyperparams[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r batch_size num_epochs mask_prompt <<< "$params"

    # Create a unique job name based on the parameters
    job_name="llama3-1b-alpaca-bs-${actual_batch_size}-nepochs-${num_epochs}-mask-${mask_prompt}"

    echo "Submitting job: ${job_name} with batch size: ${batch_size}, num epochs: ${num_epochs}, and mask prompt: ${mask_prompt}"
    sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${job_name} slurm_scripts/alpaca/llama3-1b-ablations/e2e_train.sh ${batch_size} ${num_epochs} ${mask_prompt}
done