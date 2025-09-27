"""Hyperparameter tuning with Optuna."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import structlog

from forecastit.config.settings import Settings
from forecastit.modeling.metrics import MetricsCalculator
from forecastit.modeling.registry import ModelRegistry

logger = structlog.get_logger(__name__)


class OptunaTuner:
    """Optuna-based hyperparameter tuner for forecasting models."""

    def __init__(self, settings: Settings):
        """Initialize tuner.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.metrics_calculator = MetricsCalculator()
        self.logger = logger.bind(component="optuna_tuner")

    def tune_lgbm(
        self,
        train_df: pd.DataFrame,
        feature_fn: Callable,
        objective_seconds: int = 300,
        horizon: int = 7,
        folds: int = 3,
        target_col: str = "sales",
        date_col: str = "date",
        series_cols: list[str] | None = None,
        metrics: list[str] | None = None,
        study_name: str | None = None,
    ) -> dict[str, Any]:
        """Tune LightGBM hyperparameters.

        Args:
            train_df: Training data.
            feature_fn: Function to build features.
            objective_seconds: Time limit for optimization.
            horizon: Forecast horizon.
            folds: Number of CV folds.
            target_col: Target column name.
            date_col: Date column name.
            series_cols: Series grouping columns.
            metrics: Metrics to optimize.
            study_name: Name of the study.

        Returns:
            Dictionary with tuning results.
        """
        try:
            import optuna

            from forecastit.modeling.datasets import TimeSeriesDataset
            from forecastit.modeling.ml_models import LGBMRegressorForecaster

            self.logger.info(
                "Starting LightGBM tuning",
                objective_seconds=objective_seconds,
                horizon=horizon,
                folds=folds,
            )

            if metrics is None:
                metrics = ["RMSE", "MAPE"]

            if series_cols is None:
                series_cols = ["store_id", "item_id"]

            if study_name is None:
                study_name = "lgbm_tuning"

            # Create Optuna study
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner(),
            )

            # Define objective function
            def objective(trial):
                # Suggest hyperparameters
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                }

                # Create model with suggested parameters
                model = LGBMRegressorForecaster(**params)

                # Perform cross-validation
                cv_scores = self._cross_validate_model(
                    model,
                    train_df,
                    feature_fn,
                    horizon,
                    folds,
                    target_col,
                    date_col,
                    series_cols,
                    metrics,
                )

                # Return primary metric (RMSE)
                return cv_scores.get("RMSE", float("inf"))

            # Optimize
            study.optimize(objective, timeout=objective_seconds)

            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value

            self.logger.info(
                "LightGBM tuning completed",
                best_score=best_score,
                best_params=best_params,
            )

            # Log to MLflow
            self._log_tuning_results(study, "lgbm")

            return {
                "best_params": best_params,
                "best_score": best_score,
                "study": study,
                "n_trials": len(study.trials),
            }

        except ImportError:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        except Exception as e:
            self.logger.error(f"Failed to tune LightGBM: {e}")
            raise

    def tune_sarimax(
        self,
        train_df: pd.DataFrame,
        feature_fn: Callable,
        objective_seconds: int = 180,
        horizon: int = 7,
        folds: int = 3,
        target_col: str = "sales",
        date_col: str = "date",
        series_cols: list[str] | None = None,
        metrics: list[str] | None = None,
        study_name: str | None = None,
    ) -> dict[str, Any]:
        """Tune SARIMAX hyperparameters.

        Args:
            train_df: Training data.
            feature_fn: Function to build features.
            objective_seconds: Time limit for optimization.
            horizon: Forecast horizon.
            folds: Number of CV folds.
            target_col: Target column name.
            date_col: Date column name.
            series_cols: Series grouping columns.
            metrics: Metrics to optimize.
            study_name: Name of the study.

        Returns:
            Dictionary with tuning results.
        """
        try:
            import optuna

            from forecastit.modeling.classical import SarimaxForecaster

            self.logger.info(
                "Starting SARIMAX tuning",
                objective_seconds=objective_seconds,
                horizon=horizon,
                folds=folds,
            )

            if metrics is None:
                metrics = ["RMSE", "MAPE"]

            if series_cols is None:
                series_cols = ["store_id", "item_id"]

            if study_name is None:
                study_name = "sarimax_tuning"

            # Create Optuna study
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner(),
            )

            # Define objective function
            def objective(trial):
                # Suggest hyperparameters (constrained search)
                p = trial.suggest_int("p", 0, 3)
                d = trial.suggest_int("d", 0, 2)
                q = trial.suggest_int("q", 0, 3)

                P = trial.suggest_int("P", 0, 2)
                D = trial.suggest_int("D", 0, 1)
                Q = trial.suggest_int("Q", 0, 2)
                s = trial.suggest_int("s", 7, 14)  # Weekly seasonality

                order = (p, d, q)
                seasonal_order = (P, D, Q, s)

                # Create model with suggested parameters
                model = SarimaxForecaster(
                    order=order,
                    seasonal_order=seasonal_order,
                )

                # Perform cross-validation
                cv_scores = self._cross_validate_model(
                    model,
                    train_df,
                    feature_fn,
                    horizon,
                    folds,
                    target_col,
                    date_col,
                    series_cols,
                    metrics,
                )

                # Return primary metric (RMSE)
                return cv_scores.get("RMSE", float("inf"))

            # Optimize
            study.optimize(objective, timeout=objective_seconds)

            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value

            self.logger.info(
                "SARIMAX tuning completed",
                best_score=best_score,
                best_params=best_params,
            )

            # Log to MLflow
            self._log_tuning_results(study, "sarimax")

            return {
                "best_params": best_params,
                "best_score": best_score,
                "study": study,
                "n_trials": len(study.trials),
            }

        except ImportError:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        except Exception as e:
            self.logger.error(f"Failed to tune SARIMAX: {e}")
            raise

    def tune_xgb(
        self,
        train_df: pd.DataFrame,
        feature_fn: Callable,
        objective_seconds: int = 300,
        horizon: int = 7,
        folds: int = 3,
        target_col: str = "sales",
        date_col: str = "date",
        series_cols: list[str] | None = None,
        metrics: list[str] | None = None,
        study_name: str | None = None,
    ) -> dict[str, Any]:
        """Tune XGBoost hyperparameters.

        Args:
            train_df: Training data.
            feature_fn: Function to build features.
            objective_seconds: Time limit for optimization.
            horizon: Forecast horizon.
            folds: Number of CV folds.
            target_col: Target column name.
            date_col: Date column name.
            series_cols: Series grouping columns.
            metrics: Metrics to optimize.
            study_name: Name of the study.

        Returns:
            Dictionary with tuning results.
        """
        try:
            import optuna

            from forecastit.modeling.ml_models import XGBRegressorForecaster

            self.logger.info(
                "Starting XGBoost tuning",
                objective_seconds=objective_seconds,
                horizon=horizon,
                folds=folds,
            )

            if metrics is None:
                metrics = ["RMSE", "MAPE"]

            if series_cols is None:
                series_cols = ["store_id", "item_id"]

            if study_name is None:
                study_name = "xgb_tuning"

            # Create Optuna study
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner(),
            )

            # Define objective function
            def objective(trial):
                # Suggest hyperparameters
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                }

                # Create model with suggested parameters
                model = XGBRegressorForecaster(**params)

                # Perform cross-validation
                cv_scores = self._cross_validate_model(
                    model,
                    train_df,
                    feature_fn,
                    horizon,
                    folds,
                    target_col,
                    date_col,
                    series_cols,
                    metrics,
                )

                # Return primary metric (RMSE)
                return cv_scores.get("RMSE", float("inf"))

            # Optimize
            study.optimize(objective, timeout=objective_seconds)

            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value

            self.logger.info(
                "XGBoost tuning completed",
                best_score=best_score,
                best_params=best_params,
            )

            # Log to MLflow
            self._log_tuning_results(study, "xgb")

            return {
                "best_params": best_params,
                "best_score": best_score,
                "study": study,
                "n_trials": len(study.trials),
            }

        except ImportError:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        except Exception as e:
            self.logger.error(f"Failed to tune XGBoost: {e}")
            raise

    def _cross_validate_model(
        self,
        model: Any,
        train_df: pd.DataFrame,
        feature_fn: Callable,
        horizon: int,
        folds: int,
        target_col: str,
        date_col: str,
        series_cols: list[str],
        metrics: list[str],
    ) -> dict[str, float]:
        """Perform cross-validation for a model.

        Args:
            model: Model to cross-validate.
            train_df: Training data.
            feature_fn: Function to build features.
            horizon: Forecast horizon.
            folds: Number of CV folds.
            target_col: Target column name.
            date_col: Date column name.
            series_cols: Series grouping columns.
            metrics: Metrics to calculate.

        Returns:
            Dictionary with CV scores.
        """
        from forecastit.modeling.datasets import TimeSeriesDataset

        # Create time series dataset
        dataset = TimeSeriesDataset(
            data=train_df,
            target_column=target_col,
            date_column=date_col,
            series_cols=series_cols,
        )

        # Create time series split
        tscv = dataset.create_time_series_split(n_splits=folds, test_size=horizon)

        fold_scores = []

        for train_idx, val_idx in tscv.split(train_df):
            # Split data
            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]

            # Build features
            train_features = feature_fn(train_fold)
            val_features = feature_fn(val_fold)

            # Fit model
            fitted_model = model.fit(train_features, train_features[target_col])

            # Make predictions
            predictions = fitted_model.predict(val_features)

            # Calculate metrics
            if isinstance(predictions, pd.DataFrame):
                yhat = predictions["yhat"].values
            else:
                yhat = predictions

            fold_metrics = self.metrics_calculator.calculate_metrics(
                val_fold[target_col].values,
                yhat,
                metrics=metrics,
            )

            fold_scores.append(fold_metrics)

        # Aggregate scores
        aggregated_scores = {}
        for metric in metrics:
            values = [score[metric] for score in fold_scores if metric in score]
            if values:
                aggregated_scores[metric] = np.mean(values)

        return aggregated_scores

    def _log_tuning_results(self, study: Any, model_type: str) -> None:
        """Log tuning results to MLflow.

        Args:
            study: Optuna study object.
            model_type: Type of model tuned.
        """
        try:
            registry = ModelRegistry(self.settings)

            with registry.start_run(run_name=f"tuning_{model_type}"):
                # Log best parameters
                registry.log_model_parameters(study.best_params)

                # Log best score
                registry.log_model_metrics({"best_score": study.best_value})

                # Log number of trials
                registry.log_model_metrics({"n_trials": len(study.trials)})

                self.logger.info("Logged tuning results to MLflow")

        except Exception as e:
            self.logger.warning(f"Failed to log tuning results to MLflow: {e}")


def tune_lgbm(
    train_df: pd.DataFrame,
    feature_fn: Callable,
    objective_seconds: int = 300,
    horizon: int = 7,
    folds: int = 3,
    settings: Settings | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function for LightGBM tuning.

    Args:
        train_df: Training data.
        feature_fn: Function to build features.
        objective_seconds: Time limit for optimization.
        horizon: Forecast horizon.
        folds: Number of CV folds.
        settings: Application settings.
        **kwargs: Additional arguments.

    Returns:
        Dictionary with tuning results.
    """
    if settings is None:
        settings = Settings()

    tuner = OptunaTuner(settings)

    return tuner.tune_lgbm(
        train_df=train_df,
        feature_fn=feature_fn,
        objective_seconds=objective_seconds,
        horizon=horizon,
        folds=folds,
        **kwargs,
    )


def tune_sarimax(
    train_df: pd.DataFrame,
    feature_fn: Callable,
    objective_seconds: int = 180,
    horizon: int = 7,
    folds: int = 3,
    settings: Settings | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function for SARIMAX tuning.

    Args:
        train_df: Training data.
        feature_fn: Function to build features.
        objective_seconds: Time limit for optimization.
        horizon: Forecast horizon.
        folds: Number of CV folds.
        settings: Application settings.
        **kwargs: Additional arguments.

    Returns:
        Dictionary with tuning results.
    """
    if settings is None:
        settings = Settings()

    tuner = OptunaTuner(settings)

    return tuner.tune_sarimax(
        train_df=train_df,
        feature_fn=feature_fn,
        objective_seconds=objective_seconds,
        horizon=horizon,
        folds=folds,
        **kwargs,
    )


def tune_xgb(
    train_df: pd.DataFrame,
    feature_fn: Callable,
    objective_seconds: int = 300,
    horizon: int = 7,
    folds: int = 3,
    settings: Settings | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function for XGBoost tuning.

    Args:
        train_df: Training data.
        feature_fn: Function to build features.
        objective_seconds: Time limit for optimization.
        horizon: Forecast horizon.
        folds: Number of CV folds.
        settings: Application settings.
        **kwargs: Additional arguments.

    Returns:
        Dictionary with tuning results.
    """
    if settings is None:
        settings = Settings()

    tuner = OptunaTuner(settings)

    return tuner.tune_xgb(
        train_df=train_df,
        feature_fn=feature_fn,
        objective_seconds=objective_seconds,
        horizon=horizon,
        folds=folds,
        **kwargs,
    )
