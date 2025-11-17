#!/bin/bash

# Assert that SLURM_ACC and SLURM_PARTITION are set
if [ -z "$SLURM_ACC" ] || [ -z "$SLURM_PARTITION" ]; then
    echo "SLURM_ACC and SLURM_PARTITION must be set"
    exit 1
fi

eval_root_dir=eval_results_moe/
max_gen_toks=1024

model_setups=(
    # "Qwen/Qwen3-30B-A3B qwen3-30b-a3b/intermediate lm_eval/sparsification_configs/qwen3-30b-a3b_intermediate.json 1"
    # "Qwen/Qwen3-30B-A3B qwen3-30b-a3b/input lm_eval/sparsification_configs/qwen3-30b-a3b_input.json 1"
    # "Qwen/Qwen3-30B-A3B qwen3-30b-a3b/gate lm_eval/sparsification_configs/qwen3-30b-a3b_gate.json 1"
    # "Qwen/Qwen3-30B-A3B qwen3-30b-a3b/up_proj lm_eval/sparsification_configs/qwen3-30b-a3b_up_proj.json 1"
    "Qwen/Qwen3-30B-A3B qwen3-30b-a3b/all_inputs lm_eval/sparsification_configs/qwen3-30b-a3b_all_inputs.json 1"
)
sparsity_setups=(
    # "topp 1.00"
    "topp 0.90"
    # 'topp 0.50'
)

# Loop over hyperparams and run the training scripts through slurm
for params in "${model_setups[@]}"; do
    read -r model_path model_eval_dir sparsification_config num_gpus <<< "$params"
    echo "Submitting jobs for model: ${model_path} with sparsification config: ${sparsification_config}"

    eval_dir=${eval_root_dir}/${model_eval_dir}/
    for sparsity_setup in "${sparsity_setups[@]}"; do
        read -r sparsification_rule sparsification_th <<< "$sparsity_setup"
        run_name="${sparsification_rule}_${sparsification_th}"
        # for task in arc_easy arc_challenge boolq hellaswag winogrande piqa sciq lambada triviaqa; do
        for task in arc_easy; do
            eval_job_name="eval/${model_eval_dir}/${task}/${run_name}"
            sbatch \
                -A $SLURM_ACC \
                -p $SLURM_PARTITION \
                --job-name=${eval_job_name} \
                --gres=gpu:${num_gpus} \
                --job-name=${eval_job_name} \
                slurm_scripts/harness/sparse_eval_moe.sh \
                ${model_path} \
                ${task} \
                ${eval_dir} \
                ${run_name} \
                ${sparsification_config} \
                ${sparsification_rule} \
                ${sparsification_th} \
                ${max_gen_toks} \
                ${num_gpus}
        done
    done
done