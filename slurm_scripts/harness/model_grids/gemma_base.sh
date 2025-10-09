#!/bin/bash

# Assert that SLURM_ACC and SLURM_PARTITION are set
if [ -z "$SLURM_ACC" ] || [ -z "$SLURM_PARTITION" ]; then
    echo "SLURM_ACC and SLURM_PARTITION must be set"
    exit 1
fi

eval_root_dir=eval_results_bf16/
batch_size=auto
max_gen_toks=1024

model_setups=(
    "google/gemma-3-1b-pt gemma3-1b/intermediate lm_eval/sparsification_configs/gemma3-1b_intermediate.json 1"
    "google/gemma-3-1b-pt gemma3-1b/input lm_eval/sparsification_configs/gemma3-1b_input.json 1"
    "google/gemma-3-1b-pt gemma3-1b/gate lm_eval/sparsification_configs/gemma3-1b_gate.json 1"
    "google/gemma-3-1b-pt gemma3-1b/up_proj lm_eval/sparsification_configs/gemma3-1b_up_proj.json 1"
    "google/gemma-3-1b-pt gemma3-1b/all_inputs lm_eval/sparsification_configs/gemma3-1b_all_inputs.json 1"
    "google/gemma-3-4b-pt gemma3-4b/intermediate lm_eval/sparsification_configs/gemma3-4b_intermediate.json 1"
    "google/gemma-3-4b-pt gemma3-4b/input lm_eval/sparsification_configs/gemma3-4b_input.json 1"
    "google/gemma-3-4b-pt gemma3-4b/gate lm_eval/sparsification_configs/gemma3-4b_gate.json 1"
    "google/gemma-3-4b-pt gemma3-4b/up_proj lm_eval/sparsification_configs/gemma3-4b_up_proj.json 1"
    "google/gemma-3-4b-pt gemma3-4b/all_inputs lm_eval/sparsification_configs/gemma3-4b_all_inputs.json 1"
    "google/gemma-3-12b-pt gemma3-12b/intermediate lm_eval/sparsification_configs/gemma3-12b_intermediate.json 1"
    "google/gemma-3-12b-pt gemma3-12b/input lm_eval/sparsification_configs/gemma3-12b_input.json 1"
    "google/gemma-3-12b-pt gemma3-12b/gate lm_eval/sparsification_configs/gemma3-12b_gate.json 1"
    "google/gemma-3-12b-pt gemma3-12b/up_proj lm_eval/sparsification_configs/gemma3-12b_up_proj.json 1"
    "google/gemma-3-12b-pt gemma3-12b/all_inputs lm_eval/sparsification_configs/gemma3-12b_all_inputs.json"
    "google/gemma-3-27b-pt gemma3-27b/intermediate lm_eval/sparsification_configs/gemma3-27b_intermediate.json 2"
    "google/gemma-3-27b-pt gemma3-27b/input lm_eval/sparsification_configs/gemma3-27b_input.json 2"
    "google/gemma-3-27b-pt gemma3-27b/gate lm_eval/sparsification_configs/gemma3-27b_gate.json 2"
    "google/gemma-3-27b-pt gemma3-27b/up_proj lm_eval/sparsification_configs/gemma3-27b_up_proj.json 2"
    "google/gemma-3-27b-pt gemma3-27b/all_inputs lm_eval/sparsification_configs/gemma3-27b_all_inputs.json 2"
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
    read -r model_path model_eval_dir sparsification_config num_gpus <<< "$params"
    echo "Submitting jobs for model: ${model_path} with sparsification config: ${sparsification_config}"

    eval_dir=${eval_root_dir}/${model_eval_dir}/

    for sparsity_setup in "${sparsity_setups[@]}"; do
        read -r sparsification_rule sparsification_th <<< "$sparsity_setup"

        run_name="${sparsification_rule}_${sparsification_th}"
        for task in arc_easy arc_challenge boolq hellaswag winogrande piqa sciq lambada triviaqa; do
            # Create a unique job name based on the parameters
            eval_job_name="eval/${model_eval_dir}/${task}/${run_name}"
            sbatch \
                -A $SLURM_ACC \
                -p $SLURM_PARTITION \
                --job-name=${eval_job_name} \
                --gres=gpu:${num_gpus} \
                slurm_scripts/harness/sparse_eval.sh \
                ${model_path} \
                ${task} \
                ${eval_dir} \
                ${run_name} \
                ${sparsification_config} \
                ${sparsification_rule} \
                ${sparsification_th} \
                ${batch_size} \
                ${max_gen_toks} \
                ${num_gpus}
        done
    done
done