"""
XGBoost AF Classifier
======================
Binary AF classification from HRV feature vectors.
Includes SMOTE for class imbalance, grid search, and SHAP explainability.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import pickle
import os

logger = logging.getLogger(__name__)


class XGBoostAFClassifier:
    """XGBoost-based AF classifier on HRV features."""

    def __init__(self, params: Optional[dict] = None):
        try:
            from xgboost import XGBClassifier
            self.XGBClassifier = XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed. pip install xgboost")

        self.params = params or {
            "max_depth": 6,
            "n_estimators": 200,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
        }
        self.model = None
        self.feature_names = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list = None,
        use_smote: bool = True,
        grid_search: bool = False,
    ) -> Dict[str, float]:
        """
        Train the XGBoost classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (0=Normal, 1=AF)
            feature_names: Optional feature names for explainability
            use_smote: Apply SMOTE oversampling for class balance
            grid_search: Run hyperparameter grid search

        Returns:
            Training metrics dict
        """
        self.feature_names = feature_names

        # SMOTE for class imbalance
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=42)
                X, y = sm.fit_resample(X, y)
                logger.info(f"After SMOTE: {np.sum(y==0)} Normal, {np.sum(y==1)} AF")
            except ImportError:
                logger.warning("imblearn not installed, skipping SMOTE")

        if grid_search:
            param_grid = {
                "max_depth": [4, 6, 8],
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
            }
            base = self.XGBClassifier(**self.params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            gs = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
            gs.fit(X, y)
            self.model = gs.best_estimator_
            logger.info(f"Best params: {gs.best_params_}, Best AUC: {gs.best_score_:.4f}")
        else:
            self.model = self.XGBClassifier(**self.params)
            self.model.fit(X, y)

        # Training metrics
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1": float(f1_score(y, y_pred)),
            "auroc": float(roc_auc_score(y, y_prob)),
        }
        logger.info(f"Train metrics: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict AF probability and class."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        proba = self.model.predict_proba(X)[:, 1]
        pred = self.model.predict(X)
        return pred, proba

    def get_feature_importance(self, X: np.ndarray = None) -> dict:
        """Get feature importance (built-in + optional SHAP)."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        importance = dict(zip(
            self.feature_names or [f"f{i}" for i in range(self.model.n_features_in_)],
            self.model.feature_importances_,
        ))

        if X is not None:
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X[:100])
                importance["shap_mean_abs"] = dict(zip(
                    self.feature_names or [f"f{i}" for i in range(X.shape[1])],
                    np.abs(shap_values).mean(axis=0).tolist(),
                ))
            except ImportError:
                pass

        return importance

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "features": self.feature_names, "params": self.params}, f)

    def load(self, path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["features"]
        self.params = data["params"]
