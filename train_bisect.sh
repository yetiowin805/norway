#!/bin/bash

ACCOUNT_PROJECT=    # For SBATCH account and SCRATCH_DIR
SIF_PROJECT=        # For SIF path
ACCOUNT_NAME=       # For LUMI account name

# Default values for job parameters
JOB_NAME="train_bisect"
TIME="0:30:00"
NODES=1
GPUS_PER_NODE=8
CPUS_PER_TASK=8
TRAINSET_SIZE=1000
MODEL="NorwAI/NorwAI-Mistral-7B-instruct"

# Parse command-line options
while getopts j:t:n:g:c:r:m: flag
do
    case "${flag}" in
        j) JOB_NAME=${OPTARG};;       # Job name
        t) TIME=${OPTARG};;           # Job time limit
        n) NODES=${OPTARG};;          # Number of nodes
        g) GPUS_PER_NODE=${OPTARG};;  # GPUs per node
        c) CPUS_PER_TASK=${OPTARG};;  # CPUs per task
        r) TRAINSET_SIZE=${OPTARG};;
        m) MODEL=${OPTARG};;
    esac
done

# Submit the SLURM job using a here document
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
#SBATCH --output="train_bisect_${MODEL//\//_}_%j.txt"

# Load required modules
module load LUMI PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315

# Set the path to the Singularity image
export SIF="/project/$SIF_PROJECT/EasyBuild/SW/container/PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0-dockerhash-7392c9d4dcf7.sif"

# Set Hugging Face token
export HF_TOKEN= # Add your Hugging Face token here

export SCRATCH_DIR="/scratch/$ACCOUNT_PROJECT/$ACCOUNT_NAME"
export HF_HOME="\$SCRATCH_DIR/huggingface"

# Create cache directories if they don't exist
mkdir -p "\$HF_HOME"

# Export environment variables for Singularity
export SINGULARITYENV_HF_HOME="\$HF_HOME"
export SINGULARITYENV_HF_TOKEN="\$HF_TOKEN"

# Install necessary dependencies inside the Singularity container
singularity exec --cleanenv \$SIF pip install transformers
singularity exec --cleanenv \$SIF pip install -U "huggingface_hub[cli]" torch==2.2.0+rocm5.6 torchvision==0.17.0+rocm5.6 \\
  --index-url https://download.pytorch.org/whl/rocm5.6
singularity exec --cleanenv \$SIF pip install accelerate evaluate sacrebleu sacremoses peft absl-py nltk bert_score

# Set environment variables for distributed training
export RDZV_HOST=\$(hostname)
export RDZV_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Run the training script using srun and torchrun
srun singularity exec --cleanenv --rocm --bind /users/$ACCOUNT_NAME/BiSECT:/workspace/BiSECT \\
    \$SIF torchrun --nnodes=\$SLURM_NNODES --nproc_per_node=\$SLURM_GPUS_ON_NODE --rdzv_id=\$SLURM_JOB_ID \\
    --rdzv_backend="c10d" --rdzv_endpoint="\$RDZV_HOST:\$RDZV_PORT" \\
    train_bisect.py --train_size=$TRAINSET_SIZE --model=$MODEL
EOT
