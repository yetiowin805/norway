#!/bin/bash

# Project and Account Information
# Project and Account Information
ACCOUNT_PROJECT="project_465001453"    # For SBATCH account and SCRATCH_DIR
SIF_PROJECT="project_465001410"        # For Singularity Image File (SIF) path
ACCOUNT_NAME="bungumla"                 # Username for paths

# Job Parameters with Default Values
JOB_NAME="train_norwegian"
TIME="8:00:00"
NODES=2
EPOCHS=5
GPUS_PER_NODE=8
CPUS_PER_TASK=1
TRAINSET_SIZE=10000
MODEL="NorwAI/NorwAI-Mistral-7B-instruct"
MAX_RETRIES=100


while getopts e:j:t:n:g:c:r:m: flag
do
    case "${flag}" in
        e) EPOCHS=${OPTARG};;        # Epochs
        j) JOB_NAME=${OPTARG};;       # Job name
        t) TIME=${OPTARG};;           # Job time limit
        n) NODES=${OPTARG};;          # Number of nodes
        g) GPUS_PER_NODE=${OPTARG};;  # GPUs per node
        c) CPUS_PER_TASK=${OPTARG};;  # CPUs per task
        r) TRAINSET_SIZE=${OPTARG};;  # Training set size
        m) MODEL=${OPTARG};;          # Model name
    esac
done

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$JOB_NAME
#SBATCH --account=$ACCOUNT_PROJECT
#SBATCH --time=$TIME
#SBATCH --nodes=$NODES
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=256G
#SBATCH --partition=standard-g
#SBATCH --output="train_norwegian_${MODEL//\//_}_%j.txt"

module load LUMI PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315

# Set the path to the Singularity image
export SIF="/project/$SIF_PROJECT/bungumla/EasyBuild/SW/container/PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0-dockerhash-7392c9d4dcf7.sif"

# Set Hugging Face token
export HF_TOKEN="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"

# Define Scratch and Hugging Face directories
export SCRATCH_DIR="/scratch/$ACCOUNT_PROJECT/$ACCOUNT_NAME"
export HF_HOME="\$SCRATCH_DIR/huggingface"

# Define log dir for mlworks
export LOG_DIR="/scratch/project_465001453/mlworks-logs/"

# Create cache directories if they don't exist
mkdir -p "\$HF_HOME"

# Export environment variables for Singularity
export SINGULARITYENV_HF_HOME="\$HF_HOME"
export SINGULARITYENV_HF_TOKEN="\$HF_TOKEN"
export SINGULARITYENV_LOG_DIR="\$LOG_DIR"
export SINGULARITYENV_MLFLOW_TRACKING_URI="file:/\$LOG_DIR"
export SINGULARITYENV_PYTORCH_DISABLE_FLASH_ATTENTION=1

export SINGULARITY_WITH_VENV=1

# Disabling flash attention due to ROCM error messages
export PYTORCH_DISABLE_FLASH_ATTENTION=1

singularity exec \$SIF bash -c '\$SINGULARITY_WITH_VENV; pip install transformers==4.46'

singularity exec \$SIF bash -c '\$WITH_VENV; pip install -U "huggingface_hub[cli]" torch==2.2.0+rocm5.6 torchvision==0.17.0+rocm5.6 \\
  --index-url https://download.pytorch.org/whl/rocm5.6'
singularity exec \$SIF bash -c '\$WITH_VENV; pip install accelerate evaluate sacrebleu sacremoses peft absl-py nltk bert_score mlflow'

singularity exec \$SIF bash -c '\$WITH_VENV; pip list'
singularity exec \$SIF bash -c '\$WITH_VENV; python -m site'
singularity exec \$SIF bash -c '\$WITH_VENV; which python'

export RDZV_HOST=\$(hostname)
export RDZV_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

RETRY_COUNT=0
MAX_RETRIES=$MAX_RETRIES
RETRY_WAIT=60  # Wait 60 seconds between retries

while [ \$RETRY_COUNT -lt \$MAX_RETRIES ]; do
    echo "Attempt \$((RETRY_COUNT + 1)) of \$MAX_RETRIES"

    srun singularity exec \
   --env PATH=/opt/miniconda3/envs/pytorch/bin:/user-software/venv/pytorch/bin:$PATH \
   --env PYTHONPATH=/opt/miniconda3/envs/pytorch/lib/python3.10/site-packages:/user-software/venv/pytorch/lib/python3.10/site-packages \
   --cleanenv --rocm --bind /users/$ACCOUNT_NAME/corpora/nor:/workspace/nor \\
        \$SIF torchrun --nnodes=\$SLURM_NNODES --nproc_per_node=\$SLURM_GPUS_ON_NODE --rdzv_id=\$SLURM_JOB_ID \\
        --rdzv_backend="c10d" --rdzv_endpoint="\$RDZV_HOST:\$RDZV_PORT" \\
        train_norwsynt_with_splits.py --train_size=$TRAINSET_SIZE --model=$MODEL --epochs=$EPOCHS

    EXIT_CODE=\$?

    if [ \$EXIT_CODE -eq 0 ]; then
        echo "Job completed successfully"
        exit 0
    fi

    RETRY_COUNT=\$((RETRY_COUNT + 1))

    if [ \$RETRY_COUNT -lt \$MAX_RETRIES ]; then
        echo "Job failed with exit code \$EXIT_CODE. Waiting \$RETRY_WAIT seconds before retry \$RETRY_COUNT of \$MAX_RETRIES"
        sleep \$RETRY_WAIT
    else
        echo "Job failed after \$MAX_RETRIES attempts"
        exit 1
    fi
done
EOT
