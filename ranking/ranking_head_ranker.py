"""
PyTorch-based neural ranking model for recommendation tasks.

This module implements a neural network-based learning-to-rank model using
PyTorch with support for GPU acceleration and batch training.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict


class RankingDataset(Dataset):
    """
    PyTorch Dataset for ranking tasks.
    
    Wraps feature matrix and labels for batch processing during training.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': self.X[idx],
            'label': self.y[idx]
        }


class RankingHead(nn.Module):
    """
    Neural network ranking head.
    
    A simple feedforward network that maps input features to a relevance score.
    
    Architecture:
        - Input layer: input_dim features
        - Hidden layer: 128 units with ReLU activation
        - Output layer: 1 unit (relevance score)
    
    Args:
        input_dim: Dimensionality of input features
    """
    
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Relevance scores of shape (batch_size,)
        """
        return self.fc(x).squeeze(-1)


class Ranker:
    """
    Neural ranking model trainer and predictor.
    
    This class handles data preparation, model training, and prediction
    for neural ranking tasks.
    
    Attributes:
        device: PyTorch device (CPU or CUDA)
        model: RankingHead model (None until trained)
        max_len: Maximum embedding length for padding
    """
    
    def __init__(
        self,
        max_len: int = 512,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize the ranker.
        
        Args:
            max_len: Maximum length for embedding padding
            device: PyTorch device. If None, automatically selects GPU if available
        """
        self.device = device or self._get_device()
        self.model = None
        self.max_len = max_len

    def _get_device(self) -> torch.device:
        """Get the appropriate device (GPU if available, else CPU)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(
        self,
        input_tokens: List[np.ndarray],
        output_tokens: List[List[np.ndarray]],
        scores: Optional[List[List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training or prediction.
        
        Concatenates input and output embeddings and pads to max_len.
        
        Args:
            input_tokens: List of input embeddings, one per query
            output_tokens: List of lists of candidate embeddings
            scores: Optional list of lists of relevance scores
        
        Returns:
            Tuple of (X, y) where:
            - X: Padded feature matrix of shape (n_samples, max_len)
            - y: Relevance scores of shape (n_samples,)
        """
        embeddings = []
        flat_scores = []

        for i, (input_emb, candidates) in enumerate(zip(input_tokens, output_tokens)):
            input_emb = np.array(input_emb)
            candidate_scores = scores[i] if scores is not None else [0] * len(candidates)

            for candidate_emb in candidates:
                combined_emb = np.concatenate([input_emb, candidate_emb])
                embeddings.append(combined_emb)

            flat_scores.extend(candidate_scores)

        padded_embeddings = [np.pad(e, (0, self.max_len - len(e)), 'constant') for e in embeddings]
        X = np.array(padded_embeddings)
        y = np.array(flat_scores)
        return X, y


    def train(
        self,
        input_tokens: List[np.ndarray],
        output_tokens: List[List[np.ndarray]],
        scores: List[List[int]],
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 1e-4
    ) -> None:
        """
        Train the neural ranking model.
        
        Args:
            input_tokens: Training input embeddings
            output_tokens: Training candidate embeddings
            scores: Training relevance scores
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate for Adam optimizer
        
        Examples:
            >>> ranker = Ranker(max_len=512)
            >>> ranker.train(train_input, train_output, train_scores, epochs=10)
        """
        X, y = self.prepare_data(input_tokens, output_tokens, scores)

        dataset = RankingDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = RankingHead(input_dim=self.max_len).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                scores = self.model(features)
                loss = loss_fn(scores, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

    def predict(
        self,
        input_tokens: List[np.ndarray],
        output_tokens: List[List[np.ndarray]]
    ) -> List[float]:
        """
        Generate predictions for input-output pairs.
        
        Args:
            input_tokens: Input embeddings for queries
            output_tokens: Candidate embeddings for each query
        
        Returns:
            List of prediction scores (flattened across all candidates)
        
        Examples:
            >>> predictions = ranker.predict(test_input, test_output)
        """
        X, _ = self.prepare_data(input_tokens, output_tokens, None)
        dataset = RankingDataset(X, [0]*len(X))
        dataloader = DataLoader(dataset, batch_size=64)

        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                scores = self.model(features)
                all_scores.extend(scores.cpu().numpy())

        # Group-wise top prediction or full scores
        # Best index of output tokens for each input tokens 
        return all_scores



    def evaluate_map1(
        self,
        input_tokens: List[np.ndarray],
        output_tokens: List[List[np.ndarray]],
        scores: List[List[int]]
    ) -> float:
        """
        Evaluate MAP@1 (Mean Average Precision at rank 1).
        
        Computes the proportion of queries where the top-ranked item
        is the relevant one.
        
        Args:
            input_tokens: Input embeddings
            output_tokens: Candidate embeddings (assumes 50 candidates per query)
            scores: Ground truth relevance scores
        
        Returns:
            MAP@1 score (accuracy of top-1 predictions)
        """
        preds = self.predict(input_tokens, output_tokens)
        preds = np.reshape(preds, (len(scores), 50))
        top_preds = np.argmax(preds, axis=1)
        correct = 0
        for pred, label_list in zip(top_preds, scores):
            correct += int(label_list[pred] == 1)
        return correct / len(scores)

    
def data_from_file(train_file_path, test_file_path):

    import json
    from sklearn.model_selection import train_test_split
    import os
    import pandas as pd

    # Change the working directory to the root of the GitHub repository
    notebook_dir = os.getcwd()
    if "ranking" in notebook_dir:
        os.chdir(os.path.dirname(os.path.abspath(notebook_dir)))
        print(f"Changed working directory to: {os.getcwd()}")

    # Load JSONL files
    def load_jsonl(file_path):
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]
        
    # Load data
    train_data = load_jsonl(train_file_path)
    test_data = load_jsonl(test_file_path)

    # Prepare train, validation, and test splits
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    ### change a key.
    # Update the key 'input' to 'input_token' in test_data
    for entry in test_data:
        if 'input' in entry:
            entry['input_token'] = entry.pop('input')

    def format_dataset(data):
        formatted_dataset = {
            "input_tokens": [],
            "output_tokens": [],
            "scores": []
        }
        for entry in data:
            input_tokens = entry['input_token']
            candidates = entry['candidates']
            answer = entry['answer']

            output_tokens_list = [candidate['output_tokens']+[candidate['score']] for candidate in candidates]
            scores_list = [1 if candidate['answer'] == answer else 0 for candidate in candidates]

            formatted_dataset["input_tokens"].append(input_tokens)
            formatted_dataset["output_tokens"].append(output_tokens_list)
            formatted_dataset["scores"].append(scores_list)

        return pd.DataFrame(formatted_dataset)

    df_train = format_dataset(train_data)
    df_validation = format_dataset(val_data)
    df_test = format_dataset(test_data)

    return df_train, df_validation, df_test

if __name__ == "__main__":
    df_train, df_validation, df_test = data_from_file(\
        'ranking_dataset/mistral-base-train-1108935.jsonl',\
              'ranking_dataset/mistral-base-test-1096727.jsonl')
    
    print("Training data shape:", df_train.shape)
    print("Validation data shape:", df_validation.shape)
    print("Test data shape:", df_test.shape)

    ranker = Ranker()

    # Train
    ranker.train(
        df_train['input_tokens'].tolist(),
        df_train['output_tokens'].tolist(),
        df_train['scores'].tolist(),
        epochs=100,
        batch_size=128,
    )

    # Predict
    scores = ranker.predict(
        df_test['input_tokens'].tolist(),
        df_test['output_tokens'].tolist(),
    )

    print("Scores:", scores)
    # Evaluate MAP@1
    map1 = ranker.evaluate_map1(
        df_test['input_tokens'].tolist(),
        df_test['output_tokens'].tolist(),
        df_test['scores'].tolist()
    )
    print("MAP@1:", map1)
    # Save the model
    # ranker.model.save_pretrained('ranking_model_head')
    # ranker.tokenizer.save_pretrained('ranking_model_head_tokenizer')
    # print("Model saved to 'ranking_model' directory.")