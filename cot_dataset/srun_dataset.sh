#!/bin/bash

#SBATCH --job-name=mistral_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mistral_run_%j.out
#SBATCH --gres=gpu:a100:4
#SBATCH --priority=TOP
#SBATCH --error=logs/mistral_run_%j.err



# Activate Conda environment
source ~/anaconda3/bin/activate llm_env

# Install required Python packages
pip install -U datasets transformers accelerate

echo "Starting job at $(date)"

# Run the script and capture failure
python cot_datset/cot_decoding/main.py --encode_format qa \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --max_new_tokens 256 \
    --cot_n_branches 50 \
    --decoding cot \
    --batch_size 64 \
    --data_file ./gsm8k_data/test.jsonl \
    --output_fname outputs/mistral-base-test.jsonl \
    || echo "Script failed"

echo "Job finished at $(date)"

# Deactivate Conda environment
conda deactivate
