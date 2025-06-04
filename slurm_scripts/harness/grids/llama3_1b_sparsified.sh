#!/bin/bash

eval_root_dir=sparsified_evaluation/
model_setups=(
    "meta-llama/Llama-3.2-1B llama3-1b/input lm_eval/sparsification_configs/llama3-1b_input.json"
    "meta-llama/Llama-3.2-1B llama3-1b/intermediate lm_eval/sparsification_configs/llama3-1b_intermediate.json"
    "meta-llama/Llama-3.2-1B llama3-1b/gate lm_eval/sparsification_configs/llama3-1b_gate.json"
    "meta-llama/Llama-3.2-1B llama3-1b/up_proj lm_eval/sparsification_configs/llama3-1b_up_proj.json"
)
sparsity_setups=(
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
for params in "${model_setups[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r model_path model_eval_dir sparsification_config <<< "$params"
    echo "Submitting jobs for model: ${model_path} with sparsification config: ${sparsification_config}"

    eval_dir=${eval_root_dir}/${model_eval_dir}/

    for sparsity_setup in "${sparsity_setups[@]}"; do
        read -r sparsification_rule sparsification_th <<< "$sparsity_setup"

        run_name="${sparsification_rule}_${sparsification_th}"
        # Submit evaluation jobs with dependency on the training job
        for task in arc_easy arc_challenge boolq hellaswag lambada piqa sciq triviaqa winogrande gsm8k ifeval mmlu_redux_generative; do
            # Create a unique job name based on the parameters
            eval_job_name="eval/${model_eval_dir}/${task}/${run_name}"
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 --job-name=${eval_job_name} slurm_scripts/harness/evaluate_sparsified.sh ${model_path} ${task} ${eval_dir} ${run_name} ${sparsification_config} ${sparsification_rule} ${sparsification_th}
        done
    done
done