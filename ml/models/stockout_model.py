"""
Stockout Risk Model — Predicts probability of high stockout risk (FIXED)
Course-Aligned (RandomForest + SHAP)
"""

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from ml.models.base_model import BaseModel

class StockoutRiskModel(BaseModel):
    """
    Predicts probability of HIGH stockout risk.
    Fully compatible with BaseModel (implements predict()).
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )
        self.explainer = None
        self.feature_names: list[str] = []
        self.is_trained = False

    # ============================================================
    # TRAIN
    # ============================================================
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train RF classifier and SHAP explainer."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_trained = True

        # SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

        preds = self.model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)
        print(f"✔ Stockout model trained (AUC = {auc:.4f})")

    # ============================================================
    # PREDICT
    # ============================================================
    def predict(self, X: pd.DataFrame):
        """Predict binary class (0 = low risk, 1 = high risk)."""
        X = self._check_features(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        """Predict probability of HIGH risk."""
        X = self._check_features(X.copy())
        return self.model.predict_proba(X)[:, 1]

    # ============================================================
    # HELPER (FIXED)
    # ============================================================
    def _check_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """✅ FIXED: Now returns the modified DataFrame"""
        missing = set(self.feature_names) - set(X.columns)
        for c in missing:
            X[c] = 0.0
        X = X[self.feature_names]
        return X  # ✅ FIXED: Added return statement

    # ============================================================
    # EXPLAIN (SHAP)
    # ============================================================
    def explain(self, X: pd.DataFrame):
        X = self._check_features(X)

        shap_values = self.explainer.shap_values(X)
        base_value = self.explainer.expected_value
        feature_importance = np.abs(shap_values).mean(axis=0)

        per_sample = []
        for i in range(len(X)):
            per_sample.append({
                "prediction": float(self.predict_proba(X.iloc[[i]])[0]),
                "base_value": float(base_value),
                "feature_contributions": {
                    feature: float(shap_values[i][j])
                    for j, feature in enumerate(self.feature_names)
                }
            })

        return {
            "base_value": float(base_value),
            "feature_importance": {
                self.feature_names[i]: float(feature_importance[i])
                for i in range(len(self.feature_names))
            },
            "per_sample_explanations": per_sample,
            "shap_values": shap_values.tolist()
        }
