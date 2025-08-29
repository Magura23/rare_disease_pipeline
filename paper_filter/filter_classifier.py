

from __future__ import annotations

from typing import List, Dict, Tuple, Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class RareDiseaseClassifier:
   

    def __init__(self, *, max_features: int = 50_000, ngram_range: Tuple[int, int] = (1, 2), seed: int = 42) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.seed = seed

        self.model = make_pipeline(
            TfidfVectorizer(
                lowercase=True,
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=2,
            ),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1,
            ),
        )

    @staticmethod
    def _join_title_abstract(record: Dict[str, Any]) -> str:
    
        title = record.get("title", "").strip()
        abstract = record.get("abstract", "").strip()
        return f"{title}. {abstract}".strip()




    def prepare_data( self,
    pos_records: List[Dict[str, Any]],
    neg_records: List[Dict[str, Any]],) -> Tuple[List[str], np.ndarray, List[str]]:
   
        if not pos_records or not neg_records:
            raise ValueError("Both positive and negative lists must be non-empty.")

        X_text: List[str] = []
        y_list: List[int] = []
        pmids: List[str] = []

    # positives
        for rec in pos_records:
            X_text.append(self._join_title_abstract(rec))
            y_list.append(1)
            pmids.append(rec.get("pmid", ""))

    # negatives
        for rec in neg_records:
            X_text.append(self._join_title_abstract(rec))
            y_list.append(0)
            pmids.append(rec.get("pmid", ""))

        return X_text, np.array(y_list), pmids


    def cross_validate(
        self,
        X_text: List[str],
        y: np.ndarray,
        *,
        n_splits: int = 5,
    ) -> Dict[str, List[float]]:
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        # Accumulate metrics per fold
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        f1_scores: List[float] = []

        # Loop over each fold manually to allow custom metric computation
        for train_idx, test_idx in skf.split(X_text, y):
            # Partition the data
            X_train = [X_text[i] for i in train_idx]
            y_train = y[train_idx]
            X_test = [X_text[i] for i in test_idx]
            y_test = y[test_idx]

            # Fit a fresh instance of the model on this fold
            fold_model = make_pipeline(
                TfidfVectorizer(
                    lowercase=True,
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=2,
                ),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=-1,
                ),
            )
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)

            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        scores = {
            "precision": precision_scores,
            "recall": recall_scores,
            "f1": f1_scores,
        }
        return scores

    def train(self, X_text: List[str], y: np.ndarray) -> None:
        self.model.fit(X_text, y)

    def evaluate(self, X_text: List[str], y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained classifier on a held-out set.
        """
        
        
        if not hasattr(self.model, "predict"):
            raise ValueError("The model must be trained before evaluation.")
        y_pred = self.model.predict(X_text)
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }

    def plot_cv_results(self, scores: Dict[str, List[float]]) -> None:
        """
        Plot cross–validation metrics 
        
        
        Parameters:
        
        scores : dict 
            Dictionary containing lists of per-fold metric values.  It
            must contain the keys ``precision``, ``recall`` and ``f1``.
        """
        
        metrics = ["precision", "recall", "f1"]
        for metric in metrics:
            if metric not in scores:
                raise KeyError(f"Missing metric '{metric}' in scores dictionary.")

        # Determine the number of folds 
        n_folds = len(scores["precision"])
        fold_indices = list(range(1, n_folds + 1))

        
        for metric in metrics:
            plt.figure()
            values = scores[metric]
            plt.bar(fold_indices, values)
            plt.xlabel("Fold")
            plt.ylabel(metric.capitalize())
            plt.title(f"Cross–validation {metric} per fold")
            plt.xticks(fold_indices)
            plt.ylim(0, 1)
            plt.tight_layout()
       


#__all__ = ["RareDiseaseClassifier"]