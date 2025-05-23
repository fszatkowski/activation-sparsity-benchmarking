#!/bin/bash

eval_root_dir=sparsified_evaluation/
model_setups=(
    "meta-llama/Llama-3.2-1B llama3-1b/input sparsification_configs/llama3-1b_input.json"
    "meta-llama/Llama-3.2-1B llama3-1b/intermediate sparsification_configs/llama3-1b_intermediate.json"
    "meta-llama/Llama-3.2-1B llama3-1b/gate sparsification_configs/llama3-1b_gate.json"
    "meta-llama/Llama-3.2-1B llama3-1b/up_proj sparsification_configs/llama3-1b_up_proj.json"
    # "meta-llama/Llama-3.2-3B llama3-3b/input sparsification_configs/llama3-3b_input.json"
    # "meta-llama/Llama-3.2-3B llama3-3b/intermediate sparsification_configs/llama3-3b_intermediate.json"
    # "meta-llama/Llama-3.2-3B llama3-3b/gate sparsification_configs/llama3-3b_gate.json"
    # "meta-llama/Llama-3.1-8B llama3-8b/input sparsification_configs/llama3-8b_input.json"
    # "meta-llama/Llama-3.1-8B llama3-8b/intermediate sparsification_configs/llama3-8b_intermediate.json"
    # "meta-llama/Llama-3.1-8B llama3-8b/gate sparsification_configs/llama3-8b_gate.json"
)
sparsity_setups=(
    "topk 0.01"
    "topk 0.03"
    "topk 0.05"
    "topk 0.075"
    "topk 0.1"
    "topk 0.2"
    "topk 0.25"
    "topk 0.3"
    "topk 0.35"
    "topk 0.4"
    "topk 0.45"
    "topk 0.5"
    "topk 0.6"
    "topk 0.7"
    "topk 0.8"
    "topk 1.0"
    "topp 0.999"
    "topp 0.99"
    "topp 0.98"
    "topp 0.97"
    "topp 0.95"
    "topp 0.9"
    "topp 0.85"
    "topp 0.8"
    "topp 0.75"
    "topp 0.7"
    "topp 0.6"
    "topp 0.5"
    "topp 0.4"
    "topp 0.3"
    "topp 0.2"
    "topp 0.1"
    "maxp 0.001"
    "maxp 0.003"
    "maxp 0.005"
    "maxp 0.0075"
    "maxp 0.01"
    "maxp 0.015"
    "maxp 0.02"
    "maxp 0.03"
    "maxp 0.04"
    "maxp 0.05"
    "maxp 0.06"
    "maxp 0.075"
    "maxp 0.1"
    "maxp 0.15"
    "maxp 0.2"
    "maxp 0.3"
    "maxp 0.4"
    "maxp 0.5"
    "maxp 0.75"
    "maxp 0.9"
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