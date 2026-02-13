#!/bin/bash

# Assert that SLURM_ACC and SLURM_PARTITION are set
if [ -z "$SLURM_ACC" ] || [ -z "$SLURM_PARTITION" ]; then
    echo "SLURM_ACC and SLURM_PARTITION must be set"
    exit 1
fi

eval_root_dir=eval_results_moe/
max_gen_toks=128

model_setups=(
    "allenai/OLMoE-1B-7B-0125 olmoe-7b-a1b/all_inputs lm_eval/sparsification_configs/olmoe-7b-a1b_all_inputs.json 1"
    "allenai/OLMoE-1B-7B-0125 olmoe-7b-a1b/intermediate lm_eval/sparsification_configs/olmoe-7b-a1b_intermediate.json 1"
    "allenai/OLMoE-1B-7B-0125 olmoe-7b-a1b/gate lm_eval/sparsification_configs/olmoe-7b-a1b_gate.json 1"
    "allenai/OLMoE-1B-7B-0125 olmoe-7b-a1b/input lm_eval/sparsification_configs/olmoe-7b-a1b_input.json 1"
    "allenai/OLMoE-1B-7B-0125 olmoe-7b-a1b/up_proj lm_eval/sparsification_configs/olmoe-7b-a1b_up_proj.json 1"
)
sparsity_setups=(
    "topp 1.00"
    # "topp 0.995"
    "topp 0.99"
    # "topp 0.985"
    "topp 0.98"
    # "topp 0.975"
    "topp 0.97"
    # "topp 0.96"
    "topp 0.95"
    # "topp 0.94"
    # "topp 0.93"
    "topp 0.92"
    # "topp 0.91"
    "topp 0.90"
    # "topp 0.85"
    # "topp 0.80"
    "topp 0.75"
    # "topp 0.70"
    # "topp 0.65"
    # "topp 0.60"
    'topp 0.50'
    # 'topp 0.40'
    # 'topp 0.30'
    # 'topp 0.20'
    # 'topp 0.10'
    # 'topp 0.00'
)

for task in arc_easy arc_challenge boolq winogrande piqa sciq lambada hellaswag triviaqa; do
    # Loop over hyperparams and run the training scripts through slurm
    for params in "${model_setups[@]}"; do
        read -r model_path model_eval_dir sparsification_config num_gpus <<< "$params"
        echo "Submitting jobs for model: ${model_path} with sparsification config: ${sparsification_config}"

        eval_dir=${eval_root_dir}/${model_eval_dir}/
        for sparsity_setup in "${sparsity_setups[@]}"; do
            read -r sparsification_rule sparsification_th <<< "$sparsity_setup"
            run_name="${sparsification_rule}_${sparsification_th}"
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