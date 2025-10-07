#!/bin/bash

# Assert that SLURM_ACC and SLURM_PARTITION are set
if [ -z "$SLURM_ACC" ] || [ -z "$SLURM_PARTITION" ]; then
    echo "SLURM_ACC and SLURM_PARTITION must be set"
    exit 1
fi

eval_root_dir=eval_results_effective_ranks/
batch_size=auto

model_setups=(
    "meta-llama/Llama-3.2-1B llama3-1b/input lm_eval/sparsification_configs/llama3-1b_input.json 1"
    "meta-llama/Llama-3.2-1B llama3-1b/intermediate lm_eval/sparsification_configs/llama3-1b_intermediate.json 1"
    "meta-llama/Llama-3.2-1B llama3-1b/gate lm_eval/sparsification_configs/llama3-1b_gate.json 1"
    "meta-llama/Llama-3.2-1B llama3-1b/up_proj lm_eval/sparsification_configs/llama3-1b_up_proj.json 1"
    "meta-llama/Llama-3.2-1B llama3-1b/all_inputs lm_eval/sparsification_configs/llama3-1b_all_inputs.json 1"
    "meta-llama/Llama-3.2-3B llama3-3b/input lm_eval/sparsification_configs/llama3-3b_input.json 1"
    "meta-llama/Llama-3.2-3B llama3-3b/intermediate lm_eval/sparsification_configs/llama3-3b_intermediate.json 1"
    "meta-llama/Llama-3.2-3B llama3-3b/gate lm_eval/sparsification_configs/llama3-3b_gate.json 1"
    "meta-llama/Llama-3.2-3B llama3-3b/up_proj lm_eval/sparsification_configs/llama3-3b_up_proj.json 1"
    "meta-llama/Llama-3.2-3B llama3-3b/all_inputs lm_eval/sparsification_configs/llama3-3b_all_inputs.json 1"
    "meta-llama/Llama-3.1-8B llama3-8b/input lm_eval/sparsification_configs/llama3-8b_input.json 1"
    "meta-llama/Llama-3.1-8B llama3-8b/intermediate lm_eval/sparsification_configs/llama3-8b_intermediate.json 1"
    "meta-llama/Llama-3.1-8B llama3-8b/gate lm_eval/sparsification_configs/llama3-8b_gate.json 1"
    "meta-llama/Llama-3.1-8B llama3-8b/up_proj lm_eval/sparsification_configs/llama3-8b_up_proj.json 1"
    "meta-llama/Llama-3.1-8B llama3-8b/all_inputs lm_eval/sparsification_configs/llama3-8b_all_inputs.json 1"
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
    "google/gemma-3-12b-pt gemma3-12b/all_inputs lm_eval/sparsification_configs/gemma3-12b_all_inputs.json 1"

)
sparsity_setups=(
    "topp 1.00"
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
                --gres=gpu:${num_gpus} \
                --job-name=${eval_job_name} \
                slurm_scripts/harness/sparse_eval_effective_ranks.sh \
                ${model_path} \
                ${task} \
                ${eval_dir} \
                ${run_name} \
                ${sparsification_config} \
                ${sparsification_rule} \
                ${sparsification_th} \
                ${batch_size} \
                ${num_gpus}
        done
    done
done