#!/bin/bash -l

set -euo pipefail

if [ -z "${SLURM_ACC:-}" ] || [ -z "${SLURM_PARTITION:-}" ] || [ -z "${BASE_DIR:-}" ] || [ -z "${HF_TOKEN_VAL:-}" ]; then
    echo "Error: Required environment variables are missing."
    exit 1
fi

# Path Setup
SCRIPT_PATH=$(readlink -f "$0")
GRID_DIR=$(dirname "$SCRIPT_PATH")
ABS_SCRIPTS_DIR=$(readlink -f "${GRID_DIR}/../../scripts")
BASE_DIR=$(readlink -f "${BASE_DIR}")
WORKDIR="${BASE_DIR}/intermediate"

# Resource Setup
TIME=${SLURM_TIME:-"04:00:00"}
CPUS=${SLURM_CPUS:-16}
MEM=${SLURM_MEM:-"128G"}
GPUS=${SLURM_GPUS:-1}

declare -A TASK_CMDS
HARNESS_TASKS=""  # Here add absolute path to the ../sparsified_harness/lm_eval/tasks


# Main Task Command
TASK_CMDS[humaneval]="accelerate launch ${ABS_SCRIPTS_DIR}/llada_top_p_at_diff_int.py --tasks humaneval --model llada_dist --allow_code_execution --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024,param={param},script_dir={script_dir}\""
TASK_CMDS[mbpp]="accelerate launch ${ABS_SCRIPTS_DIR}/llada_top_p_at_diff_int.py --tasks mbpp --model llada_dist --allow_code_execution --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024,param={param},script_dir={script_dir}\""
TASK_CMDS[arc_challenge]="accelerate ${ABS_SCRIPTS_DIR}/launch llada_top_p_at_diff_int.py --tasks arc_challenge --num_fewshot 0 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=False,mc_num=128,param={param}\""
TASK_CMDS[piqa]="accelerate launch ${ABS_SCRIPTS_DIR}/llada_top_p_at_diff_int.py --tasks piqa --num_fewshot 0 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=False,mc_num=128,param={param}\""
TASK_CMDS[mmlu]="accelerate launch ${ABS_SCRIPTS_DIR}/llada_top_p_at_diff_int.py --tasks mmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.0,is_check_greedy=False,mc_num=1,param={param}\""
TASK_CMDS[winogrande]="accelerate launch ${ABS_SCRIPTS_DIR}/llada_top_p_at_diff_int.py --tasks winogrande --num_fewshot 5 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.0,is_check_greedy=False,mc_num=128,param={param}\""
TASK_CMDS[ceval-valid]="accelerate launch ${ABS_SCRIPTS_DIR}/llada_top_p_at_diff_int.py --tasks ceval-valid --num_fewshot 5 --model llada_dist --batch_size 1 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.0,is_check_greedy=False,mc_num=1,param={param}\""
TASK_CMDS[cmmlu]="accelerate launch ${ABS_SCRIPTS_DIR}/llada_top_p_at_diff_int.py --tasks cmmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.0,is_check_greedy=False,mc_num=1,param={param}\""
TASK_CMDS[gpqa_main_n_shot]="accelerate launch ${ABS_SCRIPTS_DIR}/llada_top_p_at_diff_int.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=False,mc_num=128,param={param}\""
TASK_CMDS[hellaswag]="accelerate launch ${ABS_SCRIPTS_DIR}/llada_top_p_at_diff_int.py --tasks hellaswag --num_fewshot 0 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=False,mc_num=128,param={param}\""

declare -A TASK_PARAMS
TASK_PARAMS[humaneval]='60'
TASK_PARAMS[mbpp]='77.5'
TASK_PARAMS[arc_challenge]='90'
TASK_PARAMS[piqa]='60'
TASK_PARAMS[mmlu]='80'
TASK_PARAMS[winogrande]='80'
TASK_PARAMS[ceval-valid]='85'
TASK_PARAMS[cmmlu]='80'
TASK_PARAMS[gpqa_main_n_shot]='90'
TASK_PARAMS[hellaswag]='85'

GLOBAL_PARAMS='80 75 65'

GPU_IDS=$(seq -s, 0 $((GPUS-1)))
DIST_TYPE=$([ "$GPUS" -eq 1 ] && echo "NO" || echo "MULTI_GPU")

for task in "${!TASK_CMDS[@]}"; do
    PARAM_LIST="${TASK_PARAMS[$task]:-$GLOBAL_PARAMS}"
    read -ra param_array <<< "$PARAM_LIST"

    for param in "${param_array[@]}"; do
        JOB_NAME="${task}_${param}_int"
        SCRIPT_DIR="${WORKDIR}/${JOB_NAME}"
        OUT_LOG="${SCRIPT_DIR}/${JOB_NAME}.out"

        mkdir -p "${SCRIPT_DIR}"

        TASK_CMD="${TASK_CMDS[$task]}"
        TASK_CMD="${TASK_CMD//\{param\}/$param}"
        TASK_CMD="${TASK_CMD//\{script_dir\}/$SCRIPT_DIR}"

        # Prepare execution string for the compute node
        ACC_TAIL="${TASK_CMD#accelerate launch }"
        ACC_CFG="${SCRIPT_DIR}/.accelerate_config_intermediate_${JOB_NAME}.yaml"

        # We pre-build this string so the login node handles the variable expansion
        EXEC_CMD="python -m accelerate.commands.launch --config_file ${ACC_CFG} ${ACC_TAIL}"

        BATCH_SCRIPT="$(mktemp)"

        cat > "${BATCH_SCRIPT}" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUT_LOG}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --account=${SLURM_ACC}

set -euo pipefail

# 1. Environment Alignment (Escaping \$ to let compute node handle it)
conda deactivate
module purge
module load CUDA/12.4.0
module load Python/3.10.4
source ""

# 2. PYTHONPATH Precedence
export HARNESS_DIR="" # absolute path to the /sparsified_harness
export LLADA_DIR="" # absolute path to the /sparsified_llada

cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"

# Write accelerate config dynamically
cat > "${ACC_CFG}" << ACCEL_EOF
compute_environment: LOCAL_MACHINE
distributed_type: ${DIST_TYPE}
num_processes: ${GPUS}
num_machines: 1
machine_rank: 0
mixed_precision: bf16
downcast_bf16: 'no'
gpu_ids: '${GPU_IDS}'
main_training_function: main
rdzv_backend: static
same_network: true
use_cpu: false
deepspeed_config: {}
ACCEL_EOF

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN_VAL}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running Command: ${EXEC_CMD}"
eval "${EXEC_CMD}"

echo "Job Finished at \$(date)"
EOF

        # 3. Submit and print the log location back to the user
        sbatch "${BATCH_SCRIPT}"
        rm "${BATCH_SCRIPT}"
        echo "Submitted ${JOB_NAME}. Logs writing to: ${OUT_LOG}"
        echo "---"
    done
done