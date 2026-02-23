#!/bin/bash

# Assert that SLURM_ACC and SLURM_PARTITION are set
if [ -z "$SLURM_ACC" ] || [ -z "$SLURM_PARTITION" ]; then
    echo "SLURM_ACC and SLURM_PARTITION must be set"
    exit 1
fi

eval_root_dir=eval_results_thinking/
batch_size=auto
max_gen_toks=4096

model_setups=(
    "Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-instruct/input lm_eval/sparsification_configs/qwen3-4b_input.json"
    "Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-instruct/intermediate lm_eval/sparsification_configs/qwen3-4b_intermediate.json"
    "Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-instruct/gate lm_eval/sparsification_configs/qwen3-4b_gate.json"
    "Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-instruct/up_proj lm_eval/sparsification_configs/qwen3-4b_up_proj.json"
    "Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-instruct/all_inputs lm_eval/sparsification_configs/qwen3-4b_all_inputs.json"
    "Qwen/Qwen3-4B-Thinking-2507 qwen3-4b-thinking/input lm_eval/sparsification_configs/qwen3-4b_input.json"
    "Qwen/Qwen3-4B-Thinking-2507 qwen3-4b-thinking/intermediate lm_eval/sparsification_configs/qwen3-4b_intermediate.json"
    "Qwen/Qwen3-4B-Thinking-2507 qwen3-4b-thinking/gate lm_eval/sparsification_configs/qwen3-4b_gate.json"
    "Qwen/Qwen3-4B-Thinking-2507 qwen3-4b-thinking/up_proj lm_eval/sparsification_configs/qwen3-4b_up_proj.json"
    "Qwen/Qwen3-4B-Thinking-2507 qwen3-4b-thinking/all_inputs lm_eval/sparsification_configs/qwen3-4b_all_inputs.json"
)
sparsity_setups=(
    "topp 1.00"
    # "topp 0.999"
    # "topp 0.995"
    "topp 0.99"
    # "topp 0.985"
    # "topp 0.98"
    # "topp 0.975"
    # "topp 0.97"
    # "topp 0.96"
    "topp 0.95"
    # "topp 0.94"
    # "topp 0.93"
    # "topp 0.92"
    # "topp 0.91"
    "topp 0.90"
    # "topp 0.85"
    # "topp 0.80"
    # "topp 0.75"
    # "topp 0.70"
    # "topp 0.65"
    # "topp 0.60"
    # 'topp 0.50'
    # 'topp 0.40'
    # 'topp 0.30'
    # 'topp 0.20'
    # 'topp 0.10'
    # 'topp 0.00'
)
num_gpus=1

# Loop over hyperparams and run the training scripts through slurm
for params in "${model_setups[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r model_path model_eval_dir sparsification_config <<< "$params"
    echo "Submitting jobs for model: ${model_path} with sparsification config: ${sparsification_config}"

    eval_dir=${eval_root_dir}/${model_eval_dir}/

    for sparsity_setup in "${sparsity_setups[@]}"; do
        read -r sparsification_rule sparsification_th <<< "$sparsity_setup"

        run_name="${sparsification_rule}_${sparsification_th}"
        for task in mmlu_redux_generative triviaqa truthfulqa_gen gsm8k; do
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