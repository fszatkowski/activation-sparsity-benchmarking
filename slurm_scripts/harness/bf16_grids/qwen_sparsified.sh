#!/bin/bash

eval_root_dir=eval_results_bf16/
model_setups=(
    "Qwen/Qwen2.5-0.5B qwen2_5-0_5b/intermediate lm_eval/sparsification_configs/qwen2_5-0_5b_intermediate.json"
    "Qwen/Qwen2.5-0.5B qwen2_5-0_5b/input lm_eval/sparsification_configs/qwen2_5-0_5b_input.json"
    "Qwen/Qwen2.5-0.5B qwen2_5-0_5b/gate lm_eval/sparsification_configs/qwen2_5-0_5b_gate.json"
    "Qwen/Qwen2.5-0.5B qwen2_5-0_5b/up_proj lm_eval/sparsification_configs/qwen2_5-0_5b_up_proj.json"
    "Qwen/Qwen2.5-1.5B qwen2_5-1_5b/intermediate lm_eval/sparsification_configs/qwen2_5-1_5b_intermediate.json"
    "Qwen/Qwen2.5-1.5B qwen2_5-1_5b/input lm_eval/sparsification_configs/qwen2_5-1_5b_input.json"
    "Qwen/Qwen2.5-1.5B qwen2_5-1_5b/gate lm_eval/sparsification_configs/qwen2_5-1_5b_gate.json"
    "Qwen/Qwen2.5-1.5B qwen2_5-1_5b/up_proj lm_eval/sparsification_configs/qwen2_5-1_5b_up_proj.json"
    "Qwen/Qwen2.5-3B qwen2_5-3b/intermediate lm_eval/sparsification_configs/qwen2_5-3b_intermediate.json"
    "Qwen/Qwen2.5-3B qwen2_5-3b/input lm_eval/sparsification_configs/qwen2_5-3b_input.json"
    "Qwen/Qwen2.5-3B qwen2_5-3b/gate lm_eval/sparsification_configs/qwen2_5-3b_gate.json"
    "Qwen/Qwen2.5-3B qwen2_5-3b/up_proj lm_eval/sparsification_configs/qwen2_5-3b_up_proj.json"
    "Qwen/Qwen2.5-7B qwen2_5-7b/intermediate lm_eval/sparsification_configs/qwen2_5-7b_intermediate.json"
    "Qwen/Qwen2.5-7B qwen2_5-7b/input lm_eval/sparsification_configs/qwen2_5-7b_input.json"
    "Qwen/Qwen2.5-7B qwen2_5-7b/gate lm_eval/sparsification_configs/qwen2_5-7b_gate.json"
    "Qwen/Qwen2.5-7B qwen2_5-7b/up_proj lm_eval/sparsification_configs/qwen2_5-7b_up_proj.json"
    "Qwen/Qwen2.5-14B qwen2_5-14b/intermediate lm_eval/sparsification_configs/qwen2_5-14b_intermediate.json"
    "Qwen/Qwen2.5-14B qwen2_5-14b/input lm_eval/sparsification_configs/qwen2_5-14b_input.json"
    "Qwen/Qwen2.5-14B qwen2_5-14b/gate lm_eval/sparsification_configs/qwen2_5-14b_gate.json"
    "Qwen/Qwen2.5-14B qwen2_5-14b/up_proj lm_eval/sparsification_configs/qwen2_5-14b_up_proj.json"
    "Qwen/Qwen2.5-32B qwen2_5-32b/intermediate lm_eval/sparsification_configs/qwen2_5-32b_intermediate.json"
    "Qwen/Qwen2.5-32B qwen2_5-32b/input lm_eval/sparsification_configs/qwen2_5-32b_input.json"
    "Qwen/Qwen2.5-32B qwen2_5-32b/gate lm_eval/sparsification_configs/qwen2_5-32b_gate.json"
    "Qwen/Qwen2.5-32B qwen2_5-32b/up_proj lm_eval/sparsification_configs/qwen2_5-32b_up_proj.json"
    "Qwen/Qwen2.5-72B qwen2_5-72b/intermediate lm_eval/sparsification_configs/qwen2_5-72b_intermediate.json"
    "Qwen/Qwen2.5-72B qwen2_5-72b/input lm_eval/sparsification_configs/qwen2_5-72b_input.json"
    "Qwen/Qwen2.5-72B qwen2_5-72b/gate lm_eval/sparsification_configs/qwen2_5-72b_gate.json"
    "Qwen/Qwen2.5-72B qwen2_5-72b/up_proj lm_eval/sparsification_configs/qwen2_5-72b_up_proj.json"
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