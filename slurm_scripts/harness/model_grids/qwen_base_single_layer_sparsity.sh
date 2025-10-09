#!/bin/bash

# Assert that SLURM_ACC and SLURM_PARTITION are set
if [ -z "$SLURM_ACC" ] || [ -z "$SLURM_PARTITION" ]; then
    echo "SLURM_ACC and SLURM_PARTITION must be set"
    exit 1
fi

eval_root_dir=eval_results_single_layer_sparsification/
batch_size=auto
max_gen_toks=1024

model_setups=(
    "Qwen/Qwen2.5-7B qwen2_5-7b intermediate"
    "Qwen/Qwen2.5-7B qwen2_5-7b all_inputs"
    "Qwen/Qwen2.5-7B qwen2_5-7b gate"
)

layer_indices=(
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
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
    # "topp 0.92"
    # "topp 0.91"
    "topp 0.90"
    "topp 0.85"
    # "topp 0.80"
    "topp 0.75"
    # "topp 0.70"
    # "topp 0.65"
    # "topp 0.60"
    'topp 0.50'
    # 'topp 0.40'
    # 'topp 0.30'
    'topp 0.25'
    # 'topp 0.20'
    # 'topp 0.10'
    'topp 0.00'
)
num_gpus=1

# Loop over hyperparams and run the training scripts through slurm
for params in "${model_setups[@]}"; do
    # Extract batch size and num epochs from the tuple
    read -r model_path model_eval_dir sparsification_module <<< "$params"

    for layer_index in "${layer_indices[@]}"; do
        sparsification_config=lm_eval/sparsification_configs/single_layer/qwen2_5-7b_${sparsification_module}_layer_${layer_index}.json
        echo "Submitting jobs for model: ${model_path} with sparsification config: ${sparsification_config}"
        eval_dir=${eval_root_dir}/${model_eval_dir}/${sparsification_module}/${layer_index}

        for sparsity_setup in "${sparsity_setups[@]}"; do
            read -r sparsification_rule sparsification_th <<< "$sparsity_setup"

            run_name="${sparsification_module}_${layer_index}_${sparsification_rule}_${sparsification_th}"
            for task in arc_easy arc_challenge boolq hellaswag winogrande piqa sciq lambada triviaqa; do
                # Create a unique job name based on the parameters
                eval_job_name="single_layer_qwen-7b_${sparsification_module}_${layer_index}_${task}_${sparsification_rule}_${sparsification_th}"
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
done
