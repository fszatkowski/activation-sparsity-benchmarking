#!/bin/bash -l

#SBATCH --time=48:00:00   # walltime
#SBATCH --cpus-per-task=16   # number of processor cores (i.e. tasks)
#SBATCH --mem=128G   # memory (see also --mem-per-cpu)

set -e

# Assert that HF_HOME and HF_TOKEN are set
if [ -z "$HF_HOME" ] || [ -z "$HF_TOKEN" ]; then
    echo "HF_HOME and HF_TOKEN must be set"
    exit 1
fi

# Check if conda env asb exists and active it if so
# We are probably running on Athena or some other server that has conda installed
if [ -d "$SCRATCH/conda_envs/asb" ]; then
    eval "$(conda shell.bash hook)"
    conda activate asb
elif [ -d "$HOME/miniconda3/envs/asb" ]; then
    eval "$(conda shell.bash hook)"
    conda activate asb
    # If not, try to activate venv
    # We are probably running on Helios where venv has to be used instead
elif [ -d ".venv" ]; then
    module load ML-bundle/24.06a
    source .venv/bin/activate
else
    echo "Cannot activate conda env asb or venv. Exiting."
    exit 1
fi

model_dir=$1
task=$2
eval_dir=$3
run_name=$4
sparsification_config=$5
sparsification_rule=$6
sparsification_th=$7
batch_size=$8
max_gen_toks=${9:-2048}
num_gpus=${10:-1}

# Export TORCHDYNAMO_DISABLE=1 if the model name contains 'gemma'
if [[ $model_dir == *"gemma"* ]]; then
    export TORCHDYNAMO_DISABLE=1
fi

# Create a string depending on the number of GPUs by iterating from 0 to num_gpus-1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus-1)))
echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# If num_gpus is 1, we don't need to parallelize
if [ $num_gpus -eq 1 ]; then
    parallelize=False
else
    parallelize=True
fi

dtype=bfloat16
output_path=${eval_dir}/${task}/${run_name}
mkdir -p ${output_path}

python -m lm_eval.__main__ \
    --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,dtype=${dtype},parallelize=${parallelize} \
    --gen_kwargs max_gen_toks=${max_gen_toks} \
    --tasks ${task} \
    --batch_size ${batch_size} \
    --output_path ${output_path} \
    --sparsification_config ${sparsification_config} \
    --sparsification_rule ${sparsification_rule} \
    --sparsification_th_val ${sparsification_th} \
    --compute_effective_rank


