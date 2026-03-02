#!/bin/bash -l

set -euo pipefail

if [ -z "${SLURM_ACC:-}" ] || [ -z "${SLURM_PARTITION:-}" ] || [ -z "${VENV_PATH:-}" ] || [ -z "${BASE_DIR:-}" ] || [ -z "${HF_TOKEN_VAL:-}" ]; then
    echo "Error: Required environment variables are missing."
    echo "Please ensure the following are set before running: SLURM_ACC, SLURM_PARTITION, VENV_PATH, BASE_DIR, and HF_TOKEN_VAL."
    echo ""
    echo "Example usage:"
    echo "SLURM_ACC= \\" # slurm account
    echo "SLURM_PARTITION= \\" # partition of slurm
    echo "VENV_PATH= \\" # path to ../venv/bin/activate
    echo "BASE_DIR=input_intermediate \\"
    echo "HF_TOKEN_VAL= \\" # hf_your_token_here
    echo "./submit_jobs.sh"
    exit 1
fi

# Modify SLURM resources
TIME=${SLURM_TIME:-"16:00:00"}  # modify runtime as needed per task
CPUS=${SLURM_CPUS:-16}
MEM=${SLURM_MEM:-"384G"}  # adjust memory as needed, this is a general upper bound for 4 GPUs and large models, but can be reduced for smaller tasks or models. Monitor initial runs to find optimal values and avoid over-requesting resources which can lead to longer queue times.
GPUS=${SLURM_GPUS:-4}

# General setup
LIMIT=0  # limit the number of samples processed per task for quicker testing, set to 0 for no limit
WORKDIR="${BASE_DIR}/intermediate_${LIMIT}"

# Define commands per task (using {param} and {script_dir} as placeholders)
declare -A TASK_CMDS

TASK_CMDS[humaneval]="accelerate launch ../../../scripts/llada_top_p_at_diff_int.py --tasks humaneval --model llada_dist --confirm_run_unsafe_code --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024,param={param},script_dir={script_dir}\""
TASK_CMDS[mbpp]="accelerate launch ../../../scripts/llada_top_p_at_diff_int.py --tasks mbpp --model llada_dist --confirm_run_unsafe_code --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,gen_length=1024,steps=1024,block_length=1024,param={param},script_dir={script_dir}\""
TASK_CMDS[arc_challenge]="accelerate ../../../scripts/launch llada_top_p_at_diff_int.py --tasks arc_challenge --num_fewshot 0 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=False,mc_num=128,param={param}\""
TASK_CMDS[piqa]="accelerate launch ../../../scripts/llada_top_p_at_diff_int.py --tasks piqa --num_fewshot 0 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=False,mc_num=128,param={param}\""
TASK_CMDS[mmlu]="accelerate launch ../../../scripts/llada_top_p_at_diff_int.py --tasks mmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.0,is_check_greedy=False,mc_num=1,param={param}\""
TASK_CMDS[winogrande]="accelerate launch ../../../scripts/llada_top_p_at_diff_int.py --tasks winogrande --num_fewshot 5 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.0,is_check_greedy=False,mc_num=128,param={param}\""
TASK_CMDS[ceval-valid]="accelerate launch llada_top_p_at_diff_int.py --tasks ceval-valid --num_fewshot 5 --model llada_dist --batch_size 1 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.0,is_check_greedy=False,mc_num=1,param={param}\""
TASK_CMDS[cmmlu]="accelerate launch ../../../scripts/llada_top_p_at_diff_int.py --tasks cmmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.0,is_check_greedy=False,mc_num=1,param={param}\""
TASK_CMDS[gpqa_main_n_shot]="accelerate launch ../../../scripts/llada_top_p_at_diff_int.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=False,mc_num=128,param={param}\""
TASK_CMDS[hellaswag]="accelerate launch ../../../scripts/llada_top_p_at_diff_int.py --tasks hellaswag --num_fewshot 0 --model llada_dist --batch_size 8 --model_args \"model_path=GSAI-ML/LLaDA-8B-Base,cfg=0.5,is_check_greedy=False,mc_num=128,param={param}\""

# Define specific parameters mapped per task
declare -A TASK_PARAMS
TASK_PARAMS[mbpp]='80'
TASK_PARAMS[humaneval]='60'
TASK_PARAMS[arc_challenge]='90 80'
TASK_PARAMS[piqa]='60 70'

# Global fallback if a task has no specific list mapping above
GLOBAL_PARAMS='80 75 65'

# Build a dynamic comma-separated GPU ID list based on requested amount (e.g., 4 GPUs -> "0,1,2,3")
GPU_IDS=$(seq -s, 0 $((GPUS-1)))

# 5. Loop over tasks and configurations in the same fashion as your reference script
for task in "${!TASK_CMDS[@]}"; do
    # Pick parameter list for this task (task-specific or global fallback)
    PARAM_LIST="${TASK_PARAMS[$task]:-$GLOBAL_PARAMS}"

    # Convert string list to an array to handle the inner loop cleanly
    read -ra param_array <<< "$PARAM_LIST"

    for param in "${param_array[@]}"; do
        JOB_NAME="${task}_${param}_inp_int"
        SCRIPT_DIR="${WORKDIR}/${JOB_NAME}"
        LOGDIR="${SCRIPT_DIR}"
        OUT_LOG="${LOGDIR}/${JOB_NAME}.out"

        echo "Preparing job for task: ${task} | param: ${param}"
        mkdir -p "${SCRIPT_DIR}"

        # Replace placeholders {param} and {script_dir} with actual loop values
        TASK_CMD="${TASK_CMDS[$task]}"
        TASK_CMD="${TASK_CMD//\{param\}/$param}"
        TASK_CMD="${TASK_CMD//\{script_dir\}/$SCRIPT_DIR}"

        # Build runtime command by prefixing accelerate config
        ACC_TAIL="${TASK_CMD#accelerate launch }"
        ACC_CFG="${SCRIPT_DIR}/.accelerate_config_intermediate_${JOB_NAME}.yaml"
        CMD_RUNTIME="accelerate launch --config_file ${ACC_CFG} ${ACC_TAIL}"

        # Create the temporary sbatch script
        BATCH_SCRIPT="$(mktemp)"

        # NOTE: Outer EOF is unquoted so $GPUS and $GPU_IDS expand during creation.
        # Inner 'ACCEL_EOF' is quoted so YAML formatting isn't messed up by bash.
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

### MODIFY MODULES AS NEEDED FOR YOUR CLUSTER ENVIRONMENT
module purge
module load ML-bundle/24.06a
module load CUDA/12.4.0

# Activate env
source "${VENV_PATH}"

cd "${BASE_DIR}/"

# Multi-GPU visibility dynamically set
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Write accelerate config dynamically for this specific job
cat > "${ACC_CFG}" << 'ACCEL_EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
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

echo "------------------------------------------------------------"
echo "Running: ${CMD_RUNTIME}"
echo "Start time: \$(date)"
echo

# Runtime and HF cache dirs
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN_VAL}"

# Cluster / Distributed Setup
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=3

# Ensure dataset version dependency is met
pip install -U "datasets>=4.0.0"

# Execute final command
eval "${CMD_RUNTIME}"

echo
echo "End time: \$(date)"
echo "------------------------------------------------------------"
EOF

        sbatch "${BATCH_SCRIPT}"
        rm "${BATCH_SCRIPT}" # Clean up the temp file after submission
        echo "Submitted ${JOB_NAME}, log will write to: ${OUT_LOG}"
        echo "---"
    done
done