#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mistral_run_%j.out
#SBATCH --gpus-per-task=a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --error=logs/mistral_run_%j.err



# Activate Conda environment
source ~/anaconda3/bin/activate llm_env

# Install required Python packages
pip install -U datasets transformers accelerate

# Read Hugging Face token from the file
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token)

# Log in to Hugging Face
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"

echo "Starting job at $(date)"

# Run the script and capture failure
srun --ntasks=1 --cpus-per-task=4 --gpus-per-task=a100:2 --constraint=a100-80gb \
    python cot_dataset/cot_decoding/main.py --encode_format qa \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --max_new_tokens 256 \
    --cot_n_branches 50 \
    --decoding cot \
    --batch_size 64 \
    --data_file cot_dataset/cot_decoding/gsm8k_data/test.jsonl \
    --output_fname cot_dataset/cot_decoding/outputs/mistral-base-test.jsonl \
    || echo "Script failed" &

srun --ntasks=1 --cpus-per-task=4 --gpus-per-task=a100:2 --constraint=a100-80gb \
    python cot_dataset/cot_decoding/main.py --encode_format qa \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --max_new_tokens 256 \
    --cot_n_branches 50 \
    --decoding cot \
    --batch_size 64 \
    --data_file cot_dataset/cot_decoding/gsm8k_data/train.jsonl \
    --output_fname cot_dataset/cot_decoding/outputs/mistral-base-train.jsonl \
    || echo "Script failed" &

wait # Wait for all tasks to complete

echo "Job finished at $(date)"

