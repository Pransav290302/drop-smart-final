"""
ML Pipeline Service - FIXED VERSION
Uses settings.models_dir (not settings.MODELS_DIR)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from backend.app.core.config import settings

# ML model imports
from ml.models.viability_model import ViabilityModel
from ml.models.price_model import ConversionModel
from ml.models.stockout_model import StockoutRiskModel
from ml.models.clustering_model import ClusteringModel
from ml.services.price_optimizer import PriceOptimizer

# Data processing imports
from ml.data.normalization import DataNormalizer
from ml.features.engineering import engineer_features
from ml.config import get_schema_config

logger = logging.getLogger(__name__)

class MLPipelineService:
    """
    Service orchestrating:
    - Viability prediction (RandomForest)
    - Price optimization (LogisticRegression)
    - Stockout risk (RandomForest)
    - Product clustering (TF-IDF + KMeans)
    """

    def __init__(self):
        """Initialize models and services"""
        self.viability_model: Optional[ViabilityModel] = None
        self.conversion_model: Optional[ConversionModel] = None
        self.stockout_model: Optional[StockoutRiskModel] = None
        self.clustering_model: Optional[ClusteringModel] = None
        self.price_optimizer: Optional[PriceOptimizer] = None
        self.normalizer = DataNormalizer()
        # Load all models
        self._load_models()

    def _load_models(self):
        """Load all trained models from disk"""
        models_dir = Path(settings.models_dir)
        logger.info(f"Loading models from: {models_dir}")

        # 1. Load Viability Model
        try:
            viability_path = models_dir / "viability" / "model.pkl"
            if viability_path.exists():
                self.viability_model = ViabilityModel()
                self.viability_model.load(str(viability_path))
                logger.info(f"✔ Loaded model → {viability_path}")
            else:
                logger.error(f"❌ Viability model not found: {viability_path}")
        except Exception as e:
            logger.error(f"Failed to load viability model: {e}")
            self.viability_model = None

        # 2. Load Conversion Model
        try:
            conversion_path = models_dir / "price_optimizer" / "conversion_model.pkl"
            if conversion_path.exists():
                self.conversion_model = ConversionModel()
                self.conversion_model.load(str(conversion_path))
                logger.info(f"✔ Loaded model → {conversion_path}")
            else:
                logger.error(f"❌ Conversion model not found: {conversion_path}")
        except Exception as e:
            logger.error(f"Failed to load conversion model: {e}")
            self.conversion_model = None

        # 3. Load Stockout Model
        try:
            stockout_path = models_dir / "stockout_risk" / "model.pkl"
            if stockout_path.exists():
                self.stockout_model = StockoutRiskModel()
                self.stockout_model.load(str(stockout_path))
                logger.info(f"✔ Loaded model → {stockout_path}")
            else:
                logger.error(f"❌ Stockout model not found: {stockout_path}")
        except Exception as e:
            logger.error(f"Failed to load stockout model: {e}")
            self.stockout_model = None

        # 4. Load Clustering Model
        try:
            clustering_path = models_dir / "clustering" / "model.pkl"
            if clustering_path.exists():
                self.clustering_model = ClusteringModel()
                self.clustering_model.load(str(clustering_path))
                logger.info(f"✔ Loaded model → {clustering_path}")
            else:
                logger.error(f"❌ Clustering model not found: {clustering_path}")
        except Exception as e:
            logger.error(f"Failed to load clustering model: {e}")
            self.clustering_model = None

        # 5. Initialize Price Optimizer (requires conversion model)
        if self.conversion_model and self.conversion_model.is_trained:
            self.price_optimizer = PriceOptimizer(self.conversion_model)
            logger.info("✔ Initialized PriceOptimizer")
        else:
            logger.warning("⚠️ PriceOptimizer not initialized (conversion model not loaded)")
            self.price_optimizer = None

    def process_products(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process all products through ML pipeline
        Returns a list of results (one dict per product)
        """
        try:
            logger.info(f"Processing {len(df)} products through ML pipeline")

            schema = get_schema_config()
            required_features = schema["required_fields"]

            # Ensure required columns exist
            for col in required_features:
                if col not in df.columns:
                    logger.warning(f"Missing column: {col}, adding with default value 0")
                    df[col] = 0

            # Engineer features
            logger.info("Engineering features...")
            df = engineer_features(df)

            # Normalize data
            logger.info("Normalizing data...")
            df = self.normalizer.fit_transform(df)

            MODEL_FEATURES = [
                "price", "cost", "shipping_cost", "duties",
                "lead_time_days", "stock", "inventory", "quantity",
                "demand", "past_sales",
                "weight_kg", "length_cm", "width_cm", "height_cm",
                "margin", "supplier_reliability_score"
            ]

            # Ensure all model features exist
            for feat in MODEL_FEATURES:
                if feat not in df.columns:
                    logger.warning(f"Missing model feature: {feat}, adding with 0")
                    df[feat] = 0

            # Calculate margin if needed
            if "margin" not in df.columns or df["margin"].isna().all():
                df["margin"] = (
                    (df["price"] - df["cost"] - df["shipping_cost"] - df.get("duties", 0)) / df["price"]
                ).fillna(0).clip(0, 1)

            df[MODEL_FEATURES] = df[MODEL_FEATURES].fillna(0)
            X = df[MODEL_FEATURES]
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Features: {MODEL_FEATURES}")

            results = []

            for idx, row in df.iterrows():
                try:
                    result = self._process_single_product(row, X.loc[[idx]], idx)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing product {idx}: {e}", exc_info=True)
                    results.append(self._create_fallback_result(row, idx))

            results = sorted(results, key=lambda x: x["viability_score"], reverse=True)
            for rank, result in enumerate(results, 1):
                result["rank"] = rank
            logger.info(f"✅ Successfully processed {len(results)} products")
            return results

        except Exception as e:
            logger.error(f"Error in process_products: {e}", exc_info=True)
            raise

    def process_complete_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Backward compatible: returns processed results as DataFrame.
        """
        results_list = self.process_products(df)
        return pd.DataFrame(results_list)

    def _process_single_product(
        self, 
        row: pd.Series, 
        X_single: pd.DataFrame,
        idx: int
    ) -> Dict[str, Any]:
        """Process a single product through all models"""

        result = {
            "sku": str(row.get("sku", f"PRODUCT_{idx}")),
            "product_name": str(row.get("product_name", "Unknown Product")),
            "current_price": float(row.get("price", 0.0)),
            "cost": float(row.get("cost", 0.0)),
            "margin_percent": 0.0,
        }

        # ------- Robust feature check before predicting -------
        # Drop columns not used by model (if present):
        if hasattr(self.viability_model, "feature_names"):
            for c in list(X_single.columns):
                if c not in self.viability_model.feature_names:
                    X_single = X_single.drop(columns=[c])
        # Ensure not empty, not all zeros, not all NaN:
        if X_single.isnull().values.all() or (X_single.select_dtypes(include=[np.number]).sum(axis=1) == 0).all():
            logger.error(f"[{result['sku']}] All features for this row are NaN or zero. Skipping prediction!")
            raise ValueError("All input features for model are NaN or zero.")

        # 1. VIABILITY PREDICTION
        if self.viability_model and self.viability_model.is_trained:
            try:
                viability_proba = self.viability_model.predict_proba(X_single)
                viability_score = float(viability_proba[0, 1]) if hasattr(viability_proba, 'shape') and viability_proba.shape[1] > 1 else float(viability_proba[0])

                if viability_score >= 0.7:
                    viability_class = "high"
                elif viability_score >= 0.4:
                    viability_class = "medium"
                else:
                    viability_class = "low"

                result["viability_score"] = viability_score
                result["viability_class"] = viability_class

                try:
                    shap_values = self.viability_model.explain(X_single)
                    if shap_values is not None and len(shap_values) > 0:
                        feature_names = X_single.columns.tolist()
                        shap_dict = {feat: float(val) for feat, val in zip(feature_names, shap_values[0])}
                        result["shap_values"] = shap_dict
                        result["base_value"] = float(self.viability_model.explainer.expected_value)
                except Exception as e:
                    logger.debug(f"SHAP explanation failed: {e}")
                    result["shap_values"] = None
                    result["base_value"] = None

            except Exception as e:
                logger.error(f"Viability prediction failed for {result['sku']}: {e}")
                result["viability_score"] = 0.0
                result["viability_class"] = "low"
                result["shap_values"] = None
                result["base_value"] = None
        else:
            result["viability_score"] = 0.0
            result["viability_class"] = "low"
            result["shap_values"] = None
            result["base_value"] = None

        # 2. PRICE OPTIMIZATION
        if self.price_optimizer:
            try:
                current_price = result["current_price"]
                cost = result["cost"]
                optimized = self.price_optimizer.optimize_price(
                    X_single,
                    current_price=current_price,
                    min_price=cost * 1.2,
                    max_price=cost * 5.0
                )
                result["recommended_price"] = float(optimized["optimal_price"])
                recommended_price = result["recommended_price"]
                shipping = float(row.get("shipping_cost", 0.0))
                duties = float(row.get("duties", 0.0))
                margin = recommended_price - cost - shipping - duties
                margin_pct = (margin / recommended_price * 100) if recommended_price > 0 else 0.0
                result["margin_percent"] = margin_pct
            except Exception as e:
                logger.error(f"Price optimization failed for {result['sku']}: {e}")
                result["recommended_price"] = result["current_price"]
                result["margin_percent"] = 0.0
        else:
            result["recommended_price"] = result["current_price"]
            result["margin_percent"] = 0.0

        # 3. STOCKOUT RISK PREDICTION
        if self.stockout_model and self.stockout_model.is_trained:
            try:
                stockout_proba = self.stockout_model.predict_proba(X_single)
                stockout_score = float(stockout_proba[0, 1]) if hasattr(stockout_proba, 'shape') and stockout_proba.shape[1] > 1 else float(stockout_proba[0])
                if stockout_score >= 0.7:
                    risk_level = "high"
                elif stockout_score >= 0.4:
                    risk_level = "medium"
                else:
                    risk_level = "low"
                result["stockout_risk_score"] = stockout_score
                result["stockout_risk_level"] = risk_level
            except Exception as e:
                logger.error(f"Stockout prediction failed for {result['sku']}: {e}")
                result["stockout_risk_score"] = 0.0
                result["stockout_risk_level"] = "low"
        else:
            result["stockout_risk_score"] = 0.0
            result["stockout_risk_level"] = "low"

        # 4. CLUSTERING (if available)
        if self.clustering_model and self.clustering_model.is_trained:
            try:
                product_text = f"{row.get('product_name', '')} {row.get('description', '')} {row.get('category', '')}"
                cluster_id = self.clustering_model.predict([product_text])
                result["cluster_id"] = int(cluster_id[0]) if hasattr(cluster_id, '__getitem__') else int(cluster_id)
            except Exception as e:
                logger.error(f"Clustering failed for {result['sku']}: {e}")
                result["cluster_id"] = None
        else:
            result["cluster_id"] = None

        return result

    def _create_fallback_result(self, row: pd.Series, idx: int) -> Dict[str, Any]:
        """Create a fallback result when processing fails"""
        return {
            "rank": idx + 1,
            "sku": str(row.get("sku", f"PRODUCT_{idx}")),
            "product_name": str(row.get("product_name", "Unknown Product")),
            "viability_score": 0.0,
            "viability_class": "low",
            "recommended_price": float(row.get("price", 0.0)),
            "current_price": float(row.get("price", 0.0)),
            "cost": float(row.get("cost", 0.0)),
            "margin_percent": 0.0,
            "stockout_risk_score": 0.0,
            "stockout_risk_level": "low",
            "cluster_id": None,
            "shap_values": None,
            "base_value": None,
        }

    def health_check(self) -> Dict[str, bool]:
        """Check if all models are loaded and ready"""
        return {
            "viability_model": self.viability_model is not None and self.viability_model.is_trained,
            "conversion_model": self.conversion_model is not None and self.conversion_model.is_trained,
            "stockout_model": self.stockout_model is not None and self.stockout_model.is_trained,
            "clustering_model": self.clustering_model is not None and self.clustering_model.is_trained,
            "price_optimizer": self.price_optimizer is not None,
        }

# Singleton instance
_pipeline_service: Optional[MLPipelineService] = None

def get_pipeline_service() -> MLPipelineService:
    """Get or create singleton pipeline service instance"""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = MLPipelineService()
    return _pipeline_service

pipeline_service = get_pipeline_service()
