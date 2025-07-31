#!/bin/bash

eval_root_dir=eval_results_bf16/
model_setups=(
    "meta-llama/Llama-3.2-1B llama3-1b/input lm_eval/sparsification_configs/llama3-1b_input.json"
    "meta-llama/Llama-3.2-1B llama3-1b/intermediate lm_eval/sparsification_configs/llama3-1b_intermediate.json"
    "meta-llama/Llama-3.2-1B llama3-1b/gate lm_eval/sparsification_configs/llama3-1b_gate.json"
    "meta-llama/Llama-3.2-1B llama3-1b/up_proj lm_eval/sparsification_configs/llama3-1b_up_proj.json"
    "meta-llama/Llama-3.2-3B llama3-3b/input lm_eval/sparsification_configs/llama3-3b_input.json"
    "meta-llama/Llama-3.2-3B llama3-3b/intermediate lm_eval/sparsification_configs/llama3-3b_intermediate.json"
    "meta-llama/Llama-3.2-3B llama3-3b/gate lm_eval/sparsification_configs/llama3-3b_gate.json"
    "meta-llama/Llama-3.2-3B llama3-3b/up_proj lm_eval/sparsification_configs/llama3-3b_up_proj.json"
    "meta-llama/Llama-3.1-8B llama3-8b/input lm_eval/sparsification_configs/llama3-8b_input.json"
    "meta-llama/Llama-3.1-8B llama3-8b/intermediate lm_eval/sparsification_configs/llama3-8b_intermediate.json"
    "meta-llama/Llama-3.1-8B llama3-8b/gate lm_eval/sparsification_configs/llama3-8b_gate.json"
    "meta-llama/Llama-3.1-8B llama3-8b/up_proj lm_eval/sparsification_configs/llama3-8b_up_proj.json"
)

sparsity_setups=(
    "topp 1.00"
    "topp 0.995"
    "topp 0.99"
    "topp 0.985"
    "topp 0.98"
    "topp 0.975"
    "topp 0.97"
    "topp 0.96"
    "topp 0.95"
    "topp 0.94"
    "topp 0.93"
    "topp 0.92"
    "topp 0.91"
    "topp 0.90"
    "topp 0.85"
    "topp 0.80"
    "topp 0.75"
    "topp 0.70"
    "topp 0.65"
    "topp 0.60"
    'topp 0.50'
    'topp 0.40'
    'topp 0.30'
    'topp 0.20'
    'topp 0.10'
    'topp 0.00'
)

# Loop over hyperparams and run the training scripts through slurm
for params in "${model_setups[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r model_path model_eval_dir sparsification_config <<< "$params"
    echo "Submitting jobs for model: ${model_path} with sparsification config: ${sparsification_config}"

    eval_dir=${eval_root_dir}/${model_eval_dir}/

    for sparsity_setup in "${sparsity_setups[@]}"; do
        read -r sparsification_rule sparsification_th <<< "$sparsity_setup"

        run_name="${sparsification_rule}_${sparsification_th}"
        # Submit evaluation jobs with dependency on the training job
        # TODO: add triviaqa later - it's too slow
        for task in arc_easy arc_challenge boolq hellaswag winogrande piqa sciq lambada; do
            # Create a unique job name based on the parameters
            eval_job_name="eval/${model_eval_dir}/${task}/${run_name}"
            sbatch -A plgdynamic3-gpu-a100 -p plgrid-gpu-a100 --job-name=${eval_job_name} slurm_scripts/harness/evaluate_sparsified_bf16.sh ${model_path} ${task} ${eval_dir} ${run_name} ${sparsification_config} ${sparsification_rule} ${sparsification_th}
        done
    done
done