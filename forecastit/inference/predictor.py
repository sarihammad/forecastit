"""Model prediction and inference."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from forecastit.config.settings import Settings
from forecastit.modeling.registry import ModelRegistry

logger = structlog.get_logger(__name__)


class Predictor:
    """Model predictor for generating forecasts."""

    def __init__(self, settings: Settings):
        """Initialize predictor.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.logger = logger.bind(component="predictor")
        self.model_registry = ModelRegistry(settings)

        # Model components
        self.current_model = None
        self.current_model_info = None
        self.preprocessor = None
        self.feature_columns = None

    async def load_model(self, model_name: str = "forecastit_champion", stage: str = "Production") -> None:
        """Load the current champion model.

        Args:
            model_name: Name of the model to load.
            stage: Model stage.
        """
        try:
            self.logger.info("Loading model", model_name=model_name, stage=stage)

            # Try to load champion model from registry
            try:
                self.current_model, self.preprocessor = self.model_registry.load_champion(model_name)
                self.current_model_info = {
                    "model_name": model_name,
                    "stage": stage,
                    "loaded_from": "registry",
                }
                self.logger.info("Champion model loaded from registry")
            except Exception:
                # Fallback: get best model from experiments
                self.logger.info("Champion model not found in registry, falling back to best model")
                best_model = self.model_registry.get_best_model(
                    metric=self.settings.default_metric,
                    ascending=True,  # Lower is better for RMSE
                )

                if best_model is None:
                    raise ValueError("No model found to load")

                model_uri = best_model["model_uri"]
                self.current_model = self.model_registry.load_model(model_uri)
                self.current_model_info = best_model

            # Load preprocessing components
            self._load_preprocessing_components()

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error("Failed to load model", error=str(e))
            raise

    def _load_preprocessing_components(self) -> None:
        """Load preprocessing components."""
        # In a real implementation, these would be loaded from model artifacts
        # For now, we'll define them based on the feature engineering pipeline

        self.feature_columns = [
            # Calendar features
            "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
            "is_month_start", "is_month_end", "is_weekend",
            "is_holiday", "days_to_holiday", "days_from_holiday",

            # Lag features
            "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
            "sales_rolling_mean_7", "sales_rolling_mean_14", "sales_rolling_mean_28",
            "sales_rolling_std_7", "sales_rolling_std_14", "sales_rolling_std_28",
            "sales_ema_14",

            # Price/promo features
            "on_promo", "price", "price_pct_change", "promo_rolling_7",

            # Seasonality features
            "annual_sin", "annual_cos", "monthly_sin", "monthly_cos",
            "weekly_sin", "weekly_cos",
        ]

        # Simple preprocessor (in practice, this would be a fitted sklearn transformer)
        self.preprocessor = None

        self.logger.info("Preprocessing components loaded", feature_count=len(self.feature_columns))

    async def predict(
        self,
        store_id: str,
        item_id: str,
        start_date: date | datetime | str,
        end_date: date | datetime | str,
        promo_plan: list[int] | None = None,
        price_plan: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate predictions for a specific store-item combination.

        Args:
            store_id: Store identifier.
            item_id: Item identifier.
            start_date: Start date for predictions.
            end_date: End date for predictions.
            promo_plan: Promotion plan for the forecast period.
            price_plan: Price plan for the forecast period.

        Returns:
            List of prediction dictionaries.
        """
        try:
            if self.current_model is None:
                raise ValueError("Model not loaded")

            # Convert dates
            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()

            self.logger.info(
                "Generating predictions",
                store_id=store_id,
                item_id=item_id,
                start_date=start_date,
                end_date=end_date,
            )

            # Generate forecast dates
            forecast_dates = pd.date_range(start=start_date, end=end_date, freq="D")

            # Prepare features for each forecast date
            predictions = []

            for i, forecast_date in enumerate(forecast_dates):
                # Create feature vector
                features = self._create_feature_vector(
                    store_id=store_id,
                    item_id=item_id,
                    forecast_date=forecast_date,
                    promo_value=promo_plan[i] if promo_plan and i < len(promo_plan) else 0,
                    price_value=price_plan[i] if price_plan and i < len(price_plan) else None,
                )

                # Make prediction
                prediction = self._make_single_prediction(features)

                # Add to results
                predictions.append({
                    "store_id": store_id,
                    "item_id": item_id,
                    "date": forecast_date,
                    "yhat": prediction["point"],
                    "yhat_lower": prediction["lower"],
                    "yhat_upper": prediction["upper"],
                    "model_name": self.current_model_info.get("run_name", "forecastit_model") if self.current_model_info else "forecastit_model",
                    "model_version": self.current_model_info.get("run_id", "latest") if self.current_model_info else "latest",
                })

            self.logger.info(
                "Generated predictions",
                store_id=store_id,
                item_id=item_id,
                prediction_count=len(predictions),
            )

            return predictions

        except Exception as e:
            self.logger.error(
                "Prediction failed",
                store_id=store_id,
                item_id=item_id,
                error=str(e),
            )
            raise

    def _create_feature_vector(
        self,
        store_id: str,
        item_id: str,
        forecast_date: date,
        promo_value: int = 0,
        price_value: float | None = None,
    ) -> np.ndarray:
        """Create feature vector for a single prediction.

        Args:
            store_id: Store identifier.
            item_id: Item identifier.
            forecast_date: Date for the prediction.
            promo_value: Promotion value.
            price_value: Price value.

        Returns:
            Feature vector.
        """
        # Convert date to datetime for feature extraction
        dt = pd.to_datetime(forecast_date)

        # Create feature dictionary
        features = {}

        # Calendar features
        features["day_of_week"] = dt.dayofweek
        features["day_of_month"] = dt.day
        features["week_of_year"] = dt.isocalendar().week
        features["month"] = dt.month
        features["quarter"] = dt.quarter
        features["year"] = dt.year
        features["is_month_start"] = dt.is_month_start
        features["is_month_end"] = dt.is_month_end
        features["is_weekend"] = dt.dayofweek in [5, 6]

        # Holiday features (simplified)
        features["is_holiday"] = 0  # Would be calculated using holiday calendar
        features["days_to_holiday"] = 30  # Simplified
        features["days_from_holiday"] = 30  # Simplified

        # Lag features (would need historical data)
        features["sales_lag_1"] = 0.0
        features["sales_lag_7"] = 0.0
        features["sales_lag_14"] = 0.0
        features["sales_lag_28"] = 0.0

        # Rolling features
        features["sales_rolling_mean_7"] = 0.0
        features["sales_rolling_mean_14"] = 0.0
        features["sales_rolling_mean_28"] = 0.0
        features["sales_rolling_std_7"] = 0.0
        features["sales_rolling_std_14"] = 0.0
        features["sales_rolling_std_28"] = 0.0
        features["sales_ema_14"] = 0.0

        # Price/promo features
        features["on_promo"] = promo_value
        features["price"] = price_value if price_value is not None else 1.0
        features["price_pct_change"] = 0.0
        features["promo_rolling_7"] = promo_value

        # Seasonality features
        day_of_year = dt.timetuple().tm_yday
        features["annual_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        features["annual_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
        features["monthly_sin"] = np.sin(2 * np.pi * dt.month / 12)
        features["monthly_cos"] = np.cos(2 * np.pi * dt.month / 12)
        features["weekly_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        features["weekly_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

        # Convert to array in the correct order
        feature_vector = np.array([features.get(col, 0.0) for col in self.feature_columns])

        return feature_vector.reshape(1, -1)

    def _make_single_prediction(self, features: np.ndarray) -> dict[str, float]:
        """Make a single prediction with uncertainty.

        Args:
            features: Feature vector.

        Returns:
            Dictionary with point prediction and confidence intervals.
        """
        try:
            # Make prediction
            if hasattr(self.current_model, "predict"):
                predictions = self.current_model.predict(features)

                # Handle different prediction formats
                if isinstance(predictions, pd.DataFrame):
                    # Quantile model or Prophet with intervals
                    point_prediction = predictions["yhat"].iloc[0] if "yhat" in predictions.columns else predictions.iloc[0, 0]
                    lower = predictions["yhat_lower"].iloc[0] if "yhat_lower" in predictions.columns else None
                    upper = predictions["yhat_upper"].iloc[0] if "yhat_upper" in predictions.columns else None
                else:
                    # Standard point prediction
                    point_prediction = predictions[0] if len(predictions) > 0 else 0.0
                    lower = None
                    upper = None
            else:
                # Fallback for generic models
                point_prediction = 0.0
                lower = None
                upper = None

            # Generate confidence intervals if not provided
            if lower is None or upper is None:
                # Use residual-based uncertainty estimation
                uncertainty = self._estimate_uncertainty(features, point_prediction)
                lower = max(0, point_prediction - uncertainty)
                upper = point_prediction + uncertainty

            return {
                "point": max(0, point_prediction),  # Ensure non-negative
                "lower": max(0, lower),
                "upper": upper,
            }

        except Exception as e:
            self.logger.error("Single prediction failed", error=str(e))
            # Return default prediction
            return {
                "point": 1.0,
                "lower": 0.5,
                "upper": 1.5,
            }

    def _estimate_uncertainty(self, features: np.ndarray, point_prediction: float) -> float:
        """Estimate prediction uncertainty based on residual patterns.
        
        Args:
            features: Feature vector.
            point_prediction: Point prediction value.
            
        Returns:
            Estimated uncertainty (standard deviation).
        """
        # In a production system, this would use historical residuals
        # For now, use a simple heuristic based on feature complexity
        feature_complexity = np.std(features) if len(features) > 0 else 1.0
        
        # Base uncertainty on prediction magnitude and feature complexity
        base_uncertainty = abs(point_prediction) * 0.15  # 15% base uncertainty
        complexity_factor = 1.0 + feature_complexity * 0.1  # Up to 10% more uncertainty
        
        return base_uncertainty * complexity_factor

    def explain_prediction(
        self,
        store_id: str,
        item_id: str,
        forecast_date: pd.Timestamp,
        window_days: int = 14,
        top_k: int = 10,
    ) -> dict:
        """Generate SHAP-based explanation for a prediction.
        
        Args:
            store_id: Store identifier.
            item_id: Item identifier.
            forecast_date: Date to explain prediction for.
            window_days: Number of days to include in explanation window.
            top_k: Number of top features to return.
            
        Returns:
            Dictionary with explanation results.
        """
        try:
            self.logger.info(
                "Generating prediction explanation",
                store_id=store_id,
                item_id=item_id,
                forecast_date=forecast_date,
            )
            
            # Create feature vector
            features = self._create_feature_vector(
                store_id=store_id,
                item_id=item_id,
                forecast_date=forecast_date,
                promo_value=0,
                price_value=None,
            )
            
            # Generate explanation
            if self._is_tree_model():
                explanation = self._get_shap_explanation(features, top_k)
            else:
                explanation = self._get_simplified_explanation(features, top_k)
            
            # Save SHAP plot if available
            plot_path = None
            if explanation.get("shap_values") is not None:
                plot_path = self._save_shap_plot(explanation, store_id, item_id, forecast_date)
                explanation["plot_path"] = plot_path
            
            return explanation
            
        except Exception as e:
            self.logger.error(
                "Explanation generation failed",
                store_id=store_id,
                item_id=item_id,
                error=str(e),
            )
            return {
                "top_features": [],
                "feature_importance": {},
                "prediction": 0.0,
                "error": str(e),
            }

    def _is_tree_model(self) -> bool:
        """Check if current model is a tree-based model that supports SHAP."""
        if self.current_model is None:
            return False
        
        model_type = type(self.current_model).__name__.lower()
        tree_models = ['lgbmregressor', 'xgboostregressor', 'randomforestregressor', 'gradientboostingregressor']
        return any(tree_model in model_type for tree_model in tree_models)

    def _get_shap_explanation(self, features: np.ndarray, top_k: int) -> dict:
        """Get SHAP explanation for tree models."""
        try:
            import shap
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.current_model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(features)
            
            # Get feature names
            feature_names = self.feature_columns[:len(shap_values[0])]
            
            # Create feature importance mapping
            feature_importance = dict(zip(feature_names, shap_values[0]))
            
            # Get top features
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            top_features = [{"feature": name, "importance": importance} for name, importance in sorted_features[:top_k]]
            
            return {
                "top_features": top_features,
                "feature_importance": feature_importance,
                "shap_values": shap_values[0],
                "explanation_type": "shap",
            }
            
        except ImportError:
            self.logger.warning("SHAP not available, falling back to simplified explanation")
            return self._get_simplified_explanation(features, top_k)
        except Exception as e:
            self.logger.error(f"SHAP explanation failed: {e}")
            return self._get_simplified_explanation(features, top_k)

    def _get_simplified_explanation(self, features: np.ndarray, top_k: int) -> dict:
        """Get simplified explanation for non-tree models."""
        # Use feature magnitude as proxy for importance
        feature_names = self.feature_columns[:len(features[0])]
        feature_importance = dict(zip(feature_names, np.abs(features[0])))
        
        # Get top features
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = [{"feature": name, "importance": importance} for name, importance in sorted_features[:top_k]]
        
        return {
            "top_features": top_features,
            "feature_importance": feature_importance,
            "explanation_type": "simplified",
        }

    def _save_shap_plot(self, explanation: dict, store_id: str, item_id: str, forecast_date: pd.Timestamp) -> str:
        """Save SHAP beeswarm plot."""
        try:
            import matplotlib.pyplot as plt
            import shap
            
            # Create plots directory
            plots_dir = Path("forecastit/reports/explain")
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plot filename
            timestamp = forecast_date.strftime("%Y%m%d")
            plot_filename = f"shap_{store_id}_{item_id}_{timestamp}.png"
            plot_path = plots_dir / plot_filename
            
            # Create beeswarm plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use SHAP values if available
            if "shap_values" in explanation:
                shap_values = explanation["shap_values"]
                feature_names = list(explanation["feature_importance"].keys())
                
                # Create SHAP summary plot
                shap.summary_plot(
                    shap_values.reshape(1, -1),
                    feature_names=feature_names,
                    show=False,
                    max_display=15,
                )
            
            plt.title(f"SHAP Feature Importance: {store_id}/{item_id} ({forecast_date.strftime('%Y-%m-%d')})")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            self.logger.info(f"SHAP plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save SHAP plot: {e}")
            return ""

