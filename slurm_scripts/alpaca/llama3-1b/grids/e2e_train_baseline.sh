#!/bin/bash

# Create a list of tuples for batch sizes and num epochs that we will iterate over
# Tuple structure: "batch_size num_epochs"
hyperparams=(
    "16 1"
    "32 1"
    "64 1"
    "16 3"
    "32 3"
    "64 3"
    "16 5"
    "32 5"
    "64 5"
)

# Loop over hyperparams and run the training scripts through slurm
for params in "${hyperparams[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r batch_size num_epochs <<< "$params"

    # Create a unique job name based on the parameters
    job_name="llama3-1b-alpaca-bs-${actual_batch_size}-nepochs-${num_epochs}"

    echo "Submitting job: ${job_name} with batch size: ${batch_size} and num epochs: ${num_epochs}"
    sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${job_name} slurm_scripts/alpaca/llama3-1b/e2e_train.sh ${batch_size} ${num_epochs}
done