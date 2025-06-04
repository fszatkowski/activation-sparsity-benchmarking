#!/bin/bash

# Model training params
num_samples=350000 # Around 200M tokens
gacc=128 # Gradient accumulation steps
actual_batch_size=$((1*gacc)) # Batch size 2 times gradient accumulation steps

# Evaluation setup
eval_dir=results/intermediate_sparsification_kd_fineweb_200m

# Train script args: lr num_samples gradient_accumulation_steps output_dir
# Hyperparameter tuples to search over: "lr sparsity_loss sparsity_weight kd_weight kd_temperature"
hyperparams=(
    "5e-6 hoyer 0.00001 0.0 1.0"
    "5e-6 hoyer 0.00001 100.0 1.0"
    "5e-6 hoyer 0.00001 10.0 1.0"
    "5e-6 hoyer 0.00001 1.0 1.0"
    "5e-6 hoyer 0.00001 0.1 1.0"
    "5e-6 hoyer 0.00001 0.01 1.0"
    "5e-6 hoyer 0.00001 0.001 1.0"
    "5e-6 hoyer 0.00001 0.0001 1.0"
    "5e-6 hoyer 0.00001 0.00001 1.0"
    "5e-6 hoyer 0.00001 0.000001 1.0"
)

# Loop over hyperparams and run the training scripts through slurm
for params in "${hyperparams[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r lr sparsity_loss sparsity_weight kd_loss_weight kd_temperature <<< "$params"

    run_name=llama3-1b-fineweb-200m-intermediate-sparsity-loss-${sparsity_loss}-w${sparsity_weight}-lr-${lr}-kd-w-${kd_loss_weight}-temp-${kd_temperature}
    output_dir=output_models/FineWeb/Llama-3.2-1B-intermediate_kd/s${num_samples}-sparsity-loss-${sparsity_loss}-bs-${actual_batch_size}-lr-${lr}-sw-${sparsity_weight}-kd-w-${kd_loss_weight}-temp-${kd_temperature}

    # Create a unique job name based on the parameters
    job_name="llama3-1b-fineweb-s${num_samples}-lr-${lr}-${sparsity_loss}-w-${sparsity_weight}-kd-w-${kd_loss_weight}-temp-${kd_temperature}"

    # Submit the job to slurm and get it's job id
    echo "Submitting job: ${job_name}"
    job_id=$(sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${job_name} slurm_scripts/fineweb/llama3-1b/train_se_intermediate_kd.sh ${lr} ${num_samples} ${gacc} ${sparsity_loss} ${sparsity_weight} ${output_dir} ${kd_loss_weight} ${kd_temperature} | awk '{print $4}')

    # Submit evaluation jobs with dependency on the training job
    for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande gsm8k ifeval mmlu_redux_generative; do
        # Create a unique job name based on the parameters
        eval_job_name="eval_${job_name}_${task}"
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${eval_job_name} --dependency=afterok:${job_id} slurm_scripts/fineweb/llama3-1b/evaluate_harness.sh ${output_dir} ${task} ${eval_dir} ${run_name}
    done

done