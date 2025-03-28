#!/bin/bash

## run with sbatch cot_dataset/srun_dataset.sh from base directory.

#SBATCH --job-name=mistral_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mistral_run_%j.out
#SBATCH --gres=gpu:a100:2
#SBATCH --gpus-per-task=gpu:a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --error=logs/mistral_run_%j.err



# Check if Conda is installed locally in ~/anaconda3/bin
if [ -f "$HOME/anaconda3/bin/activate" ]; then
    echo "Using local Conda installation..."
    source "$HOME/anaconda3/bin/activate" llm_env
else
    echo "Local Conda not found, loading module..."
    module load conda
    source activate llm_env        # Activate Conda environment from module
fi

# Install required Python packages
pip install -U datasets transformers accelerate

# Read Hugging Face token from the file
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token)

# Log in to Hugging Face
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"

echo "Starting job at $(date)"

# Run the script and capture failure
python cot_dataset/cot_decoding/main.py --encode_format qa \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--max_new_tokens 256 \
--cot_n_branches 50 \
--decoding cot \
--batch_size 64 \
--data_file cot_dataset/cot_decoding/gsm8k_data/test.jsonl \
--output_fname cot_dataset/cot_decoding/outputs/mistral-base-test.jsonl \
|| echo "Script failed" 

echo "Job finished at $(date)"

python cot_dataset/cot_decoding/main.py --encode_format qa \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--max_new_tokens 256 \
--cot_n_branches 50 \
--decoding cot \
--batch_size 64 \
--data_file cot_dataset/cot_decoding/gsm8k_data/train.jsonl \
--output_fname cot_dataset/cot_decoding/outputs/mistral-base-train.jsonl \
|| echo "Script failed" 

echo "Job finished at $(date)"
