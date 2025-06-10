#!/bin/bash

slurm_log_dir=slurm_logs/
eval_root_dir=sparsified_evaluation_hoyer/
hoyer_weights=(
    0.0001
    0.00005
    0.00003
    0.00001
    0.000005
)
sparsification_config=lm_eval/sparsification_configs/llama3-1b_gate.json
sparsity_setups=(
    "topp 1.0"
    "topp 0.999"
    "topp 0.99"
    "topp 0.98"
    "topp 0.97"
    "topp 0.95"
    "topp 0.9"
    "topp 0.85"
    "topp 0.8"
    "topp 0.7"
    "topp 0.5"
    "topp 0.2"
    "topp 0.1"
)

# Loop over hyperparams and run the training scripts through slurm
for params in "${hoyer_weights[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r hoyer_w <<< "$params"
    model_path=output_models/FineWeb/Llama-3.2-1B-gate-preacts/s350000-sparsity-loss-hoyer-bs-128-lr-5e-6-sw-${hoyer_w}
    model_eval_dir=llama3-1b-hoyer-gate-preacts/hoyer-w-${hoyer_w}
    model_logs_dir=${slurm_log_dir}/${model_eval_dir}/

    echo "Submitting jobs for model: ${model_path} with sparsification config: ${sparsification_config}"

    eval_dir=${eval_root_dir}/${model_eval_dir}/

    for sparsity_setup in "${sparsity_setups[@]}"; do
        read -r sparsification_rule sparsification_th <<< "$sparsity_setup"

        run_name="${sparsification_rule}_${sparsification_th}"
        # Submit evaluation jobs with dependency on the training job
        # for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande gsm8k ifeval mmlu_redux_generative; do
        for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande; do
            # Create a unique job name based on the parameters
            eval_job_name="eval/${model_eval_dir}/${task}/${run_name}"
            # Log the slurm into logdir/task_name file
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${eval_job_name} --output=${model_logs_dir}/${task}_${run_name}.out slurm_scripts/harness/evaluate_sparsified.sh ${model_path} ${task} ${eval_dir} ${run_name} ${sparsification_config} ${sparsification_rule} ${sparsification_th}
        done
    done
done