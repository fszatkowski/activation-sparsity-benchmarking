#!/bin/bash

# Model training params
num_samples=350000 # Around 200M tokens
gacc=64 # Gradient accumulation steps
actual_batch_size=$((2*gacc)) # Batch size 2 times gradient accumulation steps

# Evaluation setup
eval_dir=results/gate_relufication_variants_fineweb_200m

# Create a list of tuples for batch sizes and num epochs that we will iterate over
# Train script args: lr num_samples gradient_accumulation_steps output_dir
# Hyperparameter tuples to search over: "lr"
hyperparams=(
    "5e-6 half llama3-1b-fineweb-200m-gate-relufication-hard-half"
    "5e-6 quarter llama3-1b-fineweb-200m-gate-relufication-hard-quarter"
    "5e-6 eighth llama3-1b-fineweb-200m-gate-relufication-hard-eighth"
)

# Loop over hyperparams and run the training scripts through slurm
for params in "${hyperparams[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r lr relufication_mode run_name <<< "$params"

    output_dir=output_models/FineWeb/Llama-3.2-1B-gate-relufication/s${num_samples}-hard-${relufication_mode}-bs-${actual_batch_size}-lr-${lr}

    # Create a unique job name based on the parameters
    job_name="llama3-1b-fineweb-s${num_samples}-lr-${lr}-gate-relufication-hard-${relufication_mode}-train"

    # Submit the job to slurm and get it's job id
    echo "Submitting job: ${job_name}"
    job_id=$(sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${job_name} slurm_scripts/fineweb/llama3-1b/train_relufication_gate_${relufication_mode}.sh ${lr} ${num_samples} ${gacc} hard ${output_dir} | awk '{print $4}')

    # Submit evaluation jobs with dependency on the training job
    # for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande gsm8k ifeval mmlu_redux_generative; do
    for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande; do
        # Create a unique job name based on the parameters
        eval_job_name="eval_${job_name}_${task}"
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${eval_job_name} --dependency=afterok:${job_id} slurm_scripts/harness/evaluate.sh ${output_dir} ${task} ${eval_dir} ${run_name}
    done
done
