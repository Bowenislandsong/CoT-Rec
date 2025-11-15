# CoT-Rec

Chain-of-Thought Recommendation (CoT-Rec) is a recommendation framework that leverages Chain-of-Thought reasoning to enhance decision-making by incorporating step-by-step logical reasoning into the recommendation process. This approach improves interpretability, adaptability, and performance in various recommendation scenarios.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Dataset](#dataset)
- [Testing](#testing)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)

## Overview

CoT-Rec implements advanced ranking and recommendation algorithms using Chain-of-Thought (CoT) reasoning. The framework includes:

- **CoT Decoding**: Multiple decoding strategies (greedy, beam search) for generating reasoning chains
- **Neural Ranking**: PyTorch-based ranking head for learning-to-rank tasks
- **XGBoost Ranking**: Gradient boosting-based ranker with GPU acceleration support
- **Comprehensive Metrics**: MRR, MAP, NDCG, and AP for evaluation

## Architecture

The project consists of two main components:

### 1. CoT Dataset Generation (`cot_dataset/`)
- **task.py**: Defines the GSM (Grade School Math) task format and answer extraction
- **main.py**: Main entry point for generating CoT reasoning chains
- **solve.py**: Implements different decoding strategies (greedy, CoT decoding)

### 2. Ranking Algorithms (`ranking/`)
- **data_process.py**: Utilities for embedding padding and data preparation
- **metrics.py**: Ranking evaluation metrics
- **ranking_head_ranker.py**: Neural network-based ranking model
- **xgboost_ranker.py**: XGBoost-based ranking with hyperparameter tuning

## Installation

### Using Conda (Recommended)

Create the environment from the provided configuration:

```bash
conda env create -f llm_env.yml
conda activate llm_env
```

### Manual Installation

Alternatively, install the required packages manually:

```bash
# Create a new conda environment
conda create -n cot-rec python=3.9
conda activate cot-rec

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install other dependencies
pip install transformers datasets xgboost scikit-learn pandas numpy tqdm
```

## Usage

### CoT Decoding for GSM8K

Generate Chain-of-Thought reasoning for grade school math problems:

```bash
cd cot_dataset/cot_decoding
python main.py \
    --data_file ./gsm8k_data/test.jsonl \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
    --batch_size 64 \
    --output_fname outputs/predictions.jsonl \
    --decoding cot \
    --cot_n_branches 10 \
    --cot_aggregate sum
```

### Training the Neural Ranker

Train a PyTorch-based ranking model:

```bash
cd ranking
python ranking_head_ranker.py
```

This will:
1. Load training and test data from `ranking_dataset/`
2. Train a neural ranking head for 100 epochs
3. Evaluate on test data using MAP@1
4. Report the final accuracy

### Training the XGBoost Ranker

```python
from ranking.xgboost_ranker import XGBoostRanker
import pandas as pd

# Prepare your data
ranker = XGBoostRanker(use_gpu=True)

# Train the model
ranker.train(
    input_tokens=train_df['input_tokens'].tolist(),
    output_tokens=train_df['output_tokens'].tolist(),
    scores=train_df['scores'].tolist(),
    validation_data=(val_input, val_output, val_scores)
)

# Make predictions
predictions = ranker.predict(test_input, test_output)
```

## Algorithms

### 1. Data Processing

**Padding Embeddings** (`data_process.py`):
- Pads variable-length embeddings to a uniform size
- Supports automatic target size detection
- Uses zero-padding for consistency

### 2. Ranking Metrics (`metrics.py`)

Implemented metrics for ranking evaluation:

- **MRR (Mean Reciprocal Rank)**: Measures the rank of the first correct answer
- **AP (Average Precision)**: Precision averaged across all relevant items
- **MAP (Mean Average Precision)**: Average of AP scores across queries
- **NDCG@k (Normalized Discounted Cumulative Gain)**: Position-weighted scoring metric

### 3. Neural Ranking Head (`ranking_head_ranker.py`)

PyTorch-based ranking model with:
- Two-layer feedforward network (128 hidden units)
- ReLU activation
- BCEWithLogitsLoss for training
- Support for batch training and evaluation

**Key Features**:
- Automatic GPU/CPU device selection
- Configurable embedding padding
- Batch prediction support
- MAP@1 evaluation

### 4. XGBoost Ranker (`xgboost_ranker.py`)

Gradient boosting ranker with:
- Automatic GPU detection (CUDA support)
- Pairwise ranking objective
- Early stopping with validation data
- Hyperparameter grid search

**Key Features**:
- Cross-platform support (CPU/GPU)
- Group-wise ranking
- Flexible evaluation metrics (MAP@1, NDCG)
- Model persistence

### 5. CoT Decoding Strategies

**Greedy Decoding**: 
- Standard autoregressive generation
- Deterministic single-path decoding

**CoT Decoding**:
- Beam search with multiple branches
- Confidence scoring based on probability gaps
- Three aggregation strategies:
  - `max`: Select highest-scoring candidate
  - `sum`: Aggregate scores per answer
  - `self_consistency`: Majority voting

## Dataset

The project uses two main dataset types:

### 1. GSM8K Dataset
Grade School Math problems for CoT reasoning evaluation.

### 2. Ranking Dataset
JSONL format with the following structure:

```json
{
  "input_token": [...],
  "candidates": [
    {
      "output_tokens": [...],
      "score": 0.85,
      "answer": "42"
    }
  ],
  "answer": "42"
}
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
# Test metrics
pytest tests/test_metrics.py -v

# Test data processing
pytest tests/test_data_process.py -v

# Test rankers
pytest tests/test_rankers.py -v
```

## Experimental Results

### Neural Ranking Head Performance

| Dataset | MAP@1 | Training Time | Device |
|---------|-------|---------------|--------|
| Mistral Base | TBD | ~X minutes | GPU |

### XGBoost Ranker Performance

| Dataset | NDCG | MRR | Training Time | Device |
|---------|------|-----|---------------|--------|
| Mistral Base | TBD | TBD | ~X minutes | GPU/CPU |

### CoT Decoding Results

| Strategy | Accuracy | Avg. Tokens | Inference Time |
|----------|----------|-------------|----------------|
| Greedy | TBD | TBD | TBD |
| CoT (max) | TBD | TBD | TBD |
| CoT (sum) | TBD | TBD | TBD |
| CoT (self_consistency) | TBD | TBD | TBD |

*Note: Results to be updated after running comprehensive experiments.*

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cot-rec,
  title={CoT-Rec: Chain-of-Thought Recommendation Framework},
  author={[Authors]},
  year={2024},
  url={https://github.com/Bowenislandsong/CoT-Rec}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- Built using [PyTorch](https://pytorch.org/)
- Leverages [Hugging Face Transformers](https://huggingface.co/transformers/)
- Uses [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- Inspired by Chain-of-Thought reasoning research

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
