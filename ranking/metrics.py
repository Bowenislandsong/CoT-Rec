import numpy as np

class RankingMetrics:
    def __init__(self):
        pass

    def mrr(self, ranked_list, ground_truth):
        """
        Compute Mean Reciprocal Rank (MRR).
        :param ranked_list: List of ranked answer indices (from most relevant to least relevant)
        :param ground_truth: Index of the correct answer
        :return: Reciprocal rank (1 / rank) for the first correct answer in ranked list, else 0
        """
        for rank, answer_idx in enumerate(ranked_list, 1):
            if answer_idx == ground_truth:
                return 1.0 / rank
        return 0.0

    def ap(self, ranked_list, ground_truth):
        """
        Compute Average Precision (AP).
        :param ranked_list: List of ranked answer indices (from most relevant to least relevant)
        :param ground_truth: Index of the correct answer
        :return: Average Precision
        """
        relevant_count = 0
        ap_score = 0.0
        for rank, answer_idx in enumerate(ranked_list, 1):
            if answer_idx == ground_truth:
                relevant_count += 1
                ap_score += relevant_count / rank
        return ap_score / max(1, relevant_count)

    def map(self, ranked_lists, ground_truths):
        """
        Compute Mean Average Precision (MAP).
        :param ranked_lists: List of ranked answer lists (each list corresponding to a question)
        :param ground_truths: List of ground truth indices (each corresponding to a question)
        :return: Mean Average Precision
        """
        ap_scores = []
        for ranked_list, ground_truth in zip(ranked_lists, ground_truths):
            ap_scores.append(self.ap(ranked_list, ground_truth))
        return np.mean(ap_scores)

    def ndcg(self, ranked_list, ground_truth, k=10):
        """
        Compute Normalized Discounted Cumulative Gain (NDCG) at rank k.
        :param ranked_list: List of ranked answer indices (from most relevant to least relevant)
        :param ground_truth: Index of the correct answer
        :param k: Rank cut-off (e.g., top k answers)
        :return: NDCG score
        """
        dcg = 0.0
        for rank, answer_idx in enumerate(ranked_list[:k], 1):
            if answer_idx == ground_truth:
                dcg += 1.0 / np.log2(rank + 1)
        ideal_dcg = 1.0 / np.log2(2)  # Ideal DCG when the correct answer is at rank 1
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def evaluate(self, ranked_lists, ground_truths, k=10):
        """
        Evaluate a list of ranking results based on multiple metrics.
        :param ranked_lists: List of ranked answer lists (each list corresponding to a question)
        :param ground_truths: List of ground truth indices (each corresponding to a question)
        :param k: Rank cut-off for NDCG
        :return: Dictionary of evaluation metrics
        """
        mrr_scores = [self.mrr(ranked_list, ground_truth) for ranked_list, ground_truth in zip(ranked_lists, ground_truths)]
        ap_scores = [self.ap(ranked_list, ground_truth) for ranked_list, ground_truth in zip(ranked_lists, ground_truths)]
        ndcg_scores = [self.ndcg(ranked_list, ground_truth, k) for ranked_list, ground_truth in zip(ranked_lists, ground_truths)]

        metrics = {
            "MRR": np.mean(mrr_scores),
            "AP": np.mean(ap_scores),
            "MAP": self.map(ranked_lists, ground_truths),
            "NDCG@k": np.mean(ndcg_scores)
        }
        return metrics
