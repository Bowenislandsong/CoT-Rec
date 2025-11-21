"""
XGBoost-based ranking model for recommendation tasks.

This module implements a learning-to-rank model using XGBoost with support
for GPU acceleration, hyperparameter tuning, and various ranking metrics.
"""

import numpy as np
import xgboost as xgb
import platform
import pandas as pd
from itertools import product
from copy import deepcopy
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score
from xgboost.callback import EarlyStopping
from typing import List, Tuple, Dict, Optional


class XGBoostRanker:
    """
    XGBoost-based ranking model with GPU support and hyperparameter tuning.
    
    This ranker uses gradient boosting for learning-to-rank tasks, with
    automatic GPU detection and support for group-wise ranking.
    
    Attributes:
        params: Dictionary of XGBoost parameters
        model: Trained XGBoost booster (None until trained)
        max_len: Maximum embedding length after padding
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        objective: str = 'rank:pairwise',
        eval_metric: str = 'map@1'
    ) -> None:
        """
        Initialize XGBoost ranker with automatic device selection.
        
        Args:
            use_gpu: Whether to attempt GPU usage if available
            objective: XGBoost objective function (e.g., 'rank:pairwise', 'rank:ndcg')
            eval_metric: Evaluation metric for training (e.g., 'map@1', 'ndcg')
        """
        gpu_available = use_gpu and self._detect_gpu()
        self.params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'tree_method': 'gpu_hist' if gpu_available else 'hist',
        }
        self.model = None
        self.max_len = None
        print(f"Using {'GPU' if gpu_available else 'CPU'} for training.")

    def _detect_gpu(self) -> bool:
        """
        Detect if a GPU is available and supported by XGBoost.
        
        Returns:
            True if GPU is available and supported, False otherwise
            
        Note:
            Returns False for macOS (Apple Silicon) as XGBoost doesn't
            support Apple GPU for training.
        """
        try:
            if platform.system() == "Darwin":  # macOS (M1/M2/M3/M4 chips)
                return False  # XGBoost does not support Apple's GPU for training
            from xgboost import rabit
            return True  # Assume GPU is available if XGBoost loads correctly
        except:
            return False

    def prepare_data(
        self,
        input_tokens: List[np.ndarray],
        output_tokens: List[List[np.ndarray]],
        scores: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare data from provided lists for XGBoost training.
        
        This method concatenates input and output embeddings, pads them to
        a uniform length, and creates group information for ranking.
        
        Args:
            input_tokens: List of input token embeddings, one per query
            output_tokens: List of lists of candidate output embeddings
            scores: List of lists of relevance scores for each candidate
        
        Returns:
            Tuple containing:
            - X: Padded feature matrix of shape (n_samples, max_len)
            - y: Relevance scores of shape (n_samples,)
            - group: List of group sizes (candidates per query)
        
        Examples:
            >>> ranker = XGBoostRanker(use_gpu=False)
            >>> input_tokens = [np.array([1.0, 2.0])]
            >>> output_tokens = [[np.array([3.0, 4.0]), np.array([5.0, 6.0])]]
            >>> scores = [[1, 0]]
            >>> X, y, group = ranker.prepare_data(input_tokens, output_tokens, scores)
            >>> X.shape[0] == 2  # 2 candidates
            True
        """
        embeddings = []
        flat_scores = []
        group = []

        for input_emb, candidates, candidate_scores in zip(input_tokens, output_tokens, scores):
            input_emb = np.array(input_emb)
            for candidate_emb in candidates:
                combined_emb = np.concatenate([input_emb, candidate_emb])
                embeddings.append(combined_emb)

            flat_scores.extend(candidate_scores)
            group.append(len(candidates))

        self.max_len = max(len(e) for e in embeddings)
        padded_embeddings = [np.pad(e, (0, self.max_len - len(e)), 'constant') for e in embeddings]
        X = np.array(padded_embeddings)
        y = np.array(flat_scores)
        return X, y, group

    def train(
        self,
        input_tokens: List[np.ndarray],
        output_tokens: List[List[np.ndarray]],
        scores: List[List[int]],
        validation_data: Optional[Tuple[List[np.ndarray], List[List[np.ndarray]], List[List[int]]]] = None
    ) -> xgb.Booster:
        """
        Train the XGBoost ranking model.
        
        Args:
            input_tokens: Training input token embeddings
            output_tokens: Training candidate output embeddings
            scores: Training relevance scores
            validation_data: Optional tuple of (input_tokens, output_tokens, scores)
                           for validation
        
        Returns:
            Trained XGBoost booster model
        
        Examples:
            >>> ranker = XGBoostRanker(use_gpu=False)
            >>> # Prepare training data
            >>> input_tokens = [np.random.rand(5) for _ in range(3)]
            >>> output_tokens = [[np.random.rand(4), np.random.rand(4)] for _ in range(3)]
            >>> scores = [[1, 0], [0, 1], [1, 0]]
            >>> model = ranker.train(input_tokens, output_tokens, scores)
        """
        X, y, group = self.prepare_data(input_tokens, output_tokens, scores)
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group)

        evals = []
        if validation_data:
            X_val, y_val, group_val = self.prepare_data(*validation_data)
            dval = xgb.DMatrix(X_val, label=y_val)
            dval.set_group(group_val)
            evals = [(dval, 'validation')]
        
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=1000,
                callbacks=[EarlyStopping(rounds=20, save_best=True)],
                evals=evals,
                verbose_eval=True
            )
            return self.model
        else:
            self.model = xgb.train(self.params, dtrain)
            print("Model training completed without validation.")
            return self.model

    def predict(
        self,
        input_tokens: List[np.ndarray],
        output_tokens: List[List[np.ndarray]]
    ) -> List[List[float]]:
        """
        Generate predictions for input-output pairs.
        
        Args:
            input_tokens: Input token embeddings for queries
            output_tokens: Candidate output embeddings for each query
        
        Returns:
            List of lists containing prediction scores for each candidate
        
        Raises:
            ValueError: If model has not been trained yet
        
        Examples:
            >>> # After training
            >>> predictions = ranker.predict(test_input_tokens, test_output_tokens)
            >>> len(predictions) == len(test_input_tokens)
            True
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        embeddings = []
        group_sizes = []
        for inp, candidates in zip(input_tokens, output_tokens):
            group_sizes.append(len(candidates))
            embeddings.extend(
                np.pad(np.concatenate([inp, cand]), (0, self.max_len - len(inp) - len(cand)), 'constant')
                for cand in candidates
            )

        dtest = xgb.DMatrix(np.array(embeddings))
        preds = self.model.predict(dtest)

        # Regroup predictions
        result = []
        i = 0
        for size in group_sizes:
            result.append(preds[i:i+size].tolist())
            i += size

        return result

    def hyperparameter_search(self, train_data, val_data, param_grid):
        """
        Perform grid search over hyperparameters using validation data.
        :param train_data: Tuple (input_tokens, output_tokens, scores)
        :param val_data: Tuple (input_tokens, output_tokens, scores)
        :param param_grid: Dict of lists of hyperparameter values
        :return: Best model and its metrics
        """
        keys, values = zip(*param_grid.items())
        best_score = -float("inf")
        best_model = None
        best_params = {}
        best_metrics = {}

        for combo in product(*values):
            current_params = deepcopy(self.params)
            current_params.update(dict(zip(keys, combo)))

            print(f"\nTrying params: {dict(zip(keys, combo))}")
            self.params = current_params
            self.train(*train_data, validation_data=val_data)

            val_preds = self.predict(val_data[0], val_data[1])
            true_scores_flat = [s for group in val_data[2] for s in group]
            pred_scores_flat = [s for group in val_preds for s in group]

            metrics = RankingMetrics.calculate_metrics(true_scores_flat, pred_scores_flat)
            score = metrics['mrr']  # You can change the metric used for model selection

            print(f"Validation mrr: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = deepcopy(self.model)
                best_params = current_params
                best_metrics = metrics

        print(f"\nBest Params: {best_params}")
        print(f"Best Metrics: {best_metrics}")
        self.model = best_model
        self.params = best_params
        return best_model, best_metrics

class RankingMetrics:
    @staticmethod
    def calculate_metrics(true_scores, pred_scores):
        """
        Calculate various ranking metrics.
        :param true_scores: True relevance scores.
        :param pred_scores: Predicted scores from the model.
        :return: Dictionary of metric scores.
        """
        metrics = {
            'ndcg': ndcg_score([true_scores], [pred_scores]),
            'mrr': label_ranking_average_precision_score([true_scores], [pred_scores]),
            # 'arr@1': np.average([RankingMetrics._average_reciprocal_rank_at_k(ts, ps, k=1) for ts, ps in zip(true_scores, pred_scores)]),
            # 'map@1': np.average([RankingMetrics.map_at_1(ts,ps) for ts, ps in zip(true_scores, pred_scores)]),
        }
        return metrics

    @staticmethod
    def _average_reciprocal_rank_at_k(true_scores, pred_scores, k=1):
        """
        Calculate the Average Reciprocal Rank (ARR) at rank k.
        :param true_scores: True relevance scores.
        :param pred_scores: Predicted scores from the model.
        :param k: Rank cutoff.
        :return: ARR@k score.
        """
        sorted_indices = np.argsort(pred_scores)[::-1][:k]
        for rank, idx in enumerate(sorted_indices, start=1):
            if true_scores[idx] > 0:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def map_at_1(y_true, y_score):
        """
        y_true: list of binary relevance scores, e.g., [1, 0, 0]
        y_score: list of predicted scores for the same items, e.g., [0.2, 0.6, 0.1]
        """
        top_idx = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)[0]
        return y_true[top_idx]


# Usage Example
if __name__ == "__main__":
    # Dummy dataset
    formatted_dataset = {
        "input_tokens": [np.random.rand(5) for _ in range(2)],
        "output_tokens": [[np.random.rand(4), np.random.rand(3)] for _ in range(2)],
        "scores": [[1, 0], [0, 1]]
    }

    df = pd.DataFrame(formatted_dataset)

    ranker = XGBoostRanker()
    ranker.train(
        df['input_tokens'].tolist(),
        df['output_tokens'].tolist(),
        df['scores'].tolist()
    )

    predictions = ranker.predict(
        df['input_tokens'].tolist(),
        df['output_tokens'].tolist()
    )

    true_scores_flat = [score for group in df['scores'] for score in group]
    metrics = RankingMetrics.calculate_metrics(true_scores_flat, predictions)

    print("Predictions:", predictions)
    print("Metrics:", metrics)

