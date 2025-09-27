"""Rolling origin backtesting and time-series cross-validation."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from forecastit.config.settings import Settings
from forecastit.modeling.metrics import MetricsCalculator
from forecastit.modeling.registry import ModelRegistry

logger = structlog.get_logger(__name__)


class RollingBacktester:
    """Rolling origin backtesting for time series models."""

    def __init__(self, settings: Settings):
        """Initialize backtester.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.metrics_calculator = MetricsCalculator()
        self.logger = logger.bind(component="rolling_backtester")

    def rolling_backtest(
        self,
        df: pd.DataFrame,
        series_cols: list[str],
        target_col: str,
        date_col: str,
        horizon: int,
        n_folds: int,
        fit_fn: Callable,
        predict_fn: Callable,
        feature_fn: Callable,
        metrics: list[str],
        model_name: str = "model",
        save_plots: bool = True,
    ) -> dict[str, Any]:
        """Perform rolling origin backtesting.

        Args:
            df: Input DataFrame with time series data.
            series_cols: Columns that define series (e.g., ['store_id', 'item_id']).
            target_col: Target variable column name.
            date_col: Date column name.
            horizon: Forecast horizon in days.
            n_folds: Number of backtesting folds.
            fit_fn: Function to fit model: fit_fn(train_features, train_target) -> model.
            predict_fn: Function to predict: predict_fn(model, val_features) -> predictions.
            feature_fn: Function to build features: feature_fn(df) -> features_df.
            metrics: List of metrics to calculate.
            model_name: Name of the model for logging.
            save_plots: Whether to save fold plots.

        Returns:
            Dictionary with backtesting results.
        """
        self.logger.info(
            "Starting rolling backtest",
            model_name=model_name,
            n_folds=n_folds,
            horizon=horizon,
            data_shape=df.shape,
        )

        # Sort data by series and date for strict chronological order
        sort_cols = [*series_cols, date_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Validate data is sorted chronologically
        for series_key, series_data in df.groupby(series_cols):
            series_dates = series_data[date_col]
            if not series_dates.is_monotonic_increasing:
                raise ValueError(f"Data is not sorted chronologically for series {series_key}")

        # Get date range
        min_date = df[date_col].min()
        max_date = df[date_col].max()

        # Calculate fold dates
        total_days = (max_date - min_date).days
        fold_size = total_days // n_folds

        fold_results = []
        all_predictions = []

        for fold in range(n_folds):
            self.logger.info(f"Processing fold {fold + 1}/{n_folds}")

            # Calculate fold boundaries
            fold_start_date = min_date + pd.Timedelta(days=fold * fold_size)
            fold_end_date = min_date + pd.Timedelta(days=(fold + 1) * fold_size + horizon)

            # Split data with strict chronological boundaries
            train_data = df[df[date_col] < fold_start_date + pd.Timedelta(days=fold_size)].copy()
            val_data = df[
                (df[date_col] >= fold_start_date + pd.Timedelta(days=fold_size)) &
                (df[date_col] < fold_start_date + pd.Timedelta(days=fold_size + horizon))
            ].copy()

            # STRICT VALIDATION: Ensure max(train_date) < min(val_date)
            if len(train_data) > 0 and len(val_data) > 0:
                max_train_date = train_data[date_col].max()
                min_val_date = val_data[date_col].min()
                if max_train_date >= min_val_date:
                    raise ValueError(
                        f"Fold {fold + 1}: Data leakage detected! "
                        f"max(train_date)={max_train_date} >= min(val_date)={min_val_date}"
                    )

            if len(train_data) == 0 or len(val_data) == 0:
                self.logger.warning(f"Skipping fold {fold + 1} - insufficient data")
                continue

            try:
                # Build features for training using fit_transform (learns patterns)
                train_features = feature_fn.fit_transform(train_data) if hasattr(feature_fn, 'fit_transform') else feature_fn(train_data)

                # Fit model
                model = fit_fn(train_features, train_features[target_col])

                # Build features for validation using transform only (no future lookups)
                if hasattr(feature_fn, 'transform'):
                    val_features = feature_fn.transform(val_data)
                else:
                    val_features = feature_fn(val_data)

                # Make predictions
                predictions = predict_fn(model, val_features)

                # Calculate metrics
                fold_metrics = self._calculate_fold_metrics(
                    val_data[target_col].values,
                    predictions,
                    metrics,
                )

                fold_result = {
                    "fold": fold + 1,
                    "train_start": train_data[date_col].min(),
                    "train_end": train_data[date_col].max(),
                    "val_start": val_data[date_col].min(),
                    "val_end": val_data[date_col].max(),
                    "metrics": fold_metrics,
                    "predictions": predictions,
                    "actual": val_data[target_col].values,
                    "dates": val_data[date_col].values,
                }

                fold_results.append(fold_result)
                all_predictions.extend(predictions)

                # Save fold plot if requested
                if save_plots:
                    self._save_fold_plot(fold_result, model_name, fold + 1)

                self.logger.info(
                    f"Completed fold {fold + 1}",
                    rmse=fold_metrics.get("RMSE", 0),
                    mape=fold_metrics.get("MAPE", 0),
                )

            except Exception as e:
                self.logger.error(f"Failed fold {fold + 1}: {e}")
                continue

        if not fold_results:
            raise ValueError("No successful folds completed")

        # Aggregate results
        aggregated_results = self._aggregate_results(fold_results, metrics)

        # Save summary plot
        if save_plots:
            self._save_summary_plot(fold_results, model_name)

        # Log to MLflow if available
        self._log_to_mlflow(fold_results, aggregated_results, model_name)

        self.logger.info(
            "Rolling backtest completed",
            model_name=model_name,
            successful_folds=len(fold_results),
            aggregated_metrics=aggregated_results,
        )

        return {
            "model_name": model_name,
            "fold_results": fold_results,
            "aggregated_metrics": aggregated_results,
            "total_predictions": len(all_predictions),
            "successful_folds": len(fold_results),
        }

    def _calculate_fold_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        metrics: list[str],
        predicted_lower: np.ndarray | None = None,
        predicted_upper: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Calculate metrics for a single fold.

        Args:
            actual: Actual values.
            predicted: Predicted values.
            metrics: List of metrics to calculate.
            predicted_lower: Lower bound predictions for quantile metrics.
            predicted_upper: Upper bound predictions for quantile metrics.

        Returns:
            Dictionary with metric values.
        """
        # Handle quantile predictions
        if isinstance(predicted, dict) and 'quantiles' in predicted:
            # Extract quantile predictions
            quantile_preds = predicted['quantiles']
            predicted_array = quantile_preds.get(0.5, predicted_array)  # Use median as point prediction
            
            # Calculate pinball losses if quantiles are available
            quantile_metrics = {}
            for alpha in [0.1, 0.5, 0.9]:
                if alpha in quantile_preds:
                    quantile_metrics[f'pinball_{alpha}'] = self._calculate_pinball_loss(
                        actual, quantile_preds[alpha], alpha
                    )
        else:
            predicted_array = predicted
            quantile_metrics = {}

        # Calculate standard metrics
        standard_metrics = self.metrics_calculator.calculate_metrics(
            actual,
            predicted_array,
            metrics=metrics,
        )

        # Combine all metrics
        all_metrics = {**standard_metrics, **quantile_metrics}
        
        # Validate MASE is finite and non-NaN
        if 'mase' in all_metrics:
            if not np.isfinite(all_metrics['mase']):
                self.logger.warning(f"MASE is not finite: {all_metrics['mase']}")
                all_metrics['mase'] = np.nan

        return all_metrics

    def _calculate_pinball_loss(self, actual: np.ndarray, predicted: np.ndarray, alpha: float) -> float:
        """Calculate pinball loss for quantile predictions.
        
        Args:
            actual: Actual values.
            predicted: Quantile predictions.
            alpha: Quantile level (0 < alpha < 1).
            
        Returns:
            Pinball loss value.
        """
        errors = actual - predicted
        return np.mean(np.maximum(alpha * errors, (alpha - 1) * errors))

    def _aggregate_results(
        self,
        fold_results: list[dict],
        metrics: list[str],
    ) -> dict[str, float]:
        """Aggregate metrics across folds.

        Args:
            fold_results: List of fold results.
            metrics: List of metrics.

        Returns:
            Dictionary with aggregated metrics.
        """
        aggregated = {}

        for metric in metrics:
            values = [fold["metrics"][metric] for fold in fold_results if metric in fold["metrics"]]
            if values:
                aggregated[metric] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)

        return aggregated

    def _save_fold_plot(
        self,
        fold_result: dict,
        model_name: str,
        fold: int,
    ) -> None:
        """Save plot for a single fold.

        Args:
            fold_result: Results for the fold.
            model_name: Name of the model.
            fold: Fold number.
        """
        try:
            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
            except ImportError:
                sns = None

            # Set style
            if sns:
                plt.style.use("seaborn-v0_8")

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))

            dates = fold_result["dates"]
            actual = fold_result["actual"]
            predictions = fold_result["predictions"]

            # Plot actual vs predicted
            ax.plot(dates, actual, label="Actual", linewidth=2)
            ax.plot(dates, predictions, label="Predicted", linewidth=2, linestyle="--")

            # Add confidence intervals if available
            if isinstance(predictions, dict) and "yhat_lower" in predictions:
                ax.fill_between(
                    dates,
                    predictions["yhat_lower"],
                    predictions["yhat_upper"],
                    alpha=0.3,
                    label="Confidence Interval",
                )

            ax.set_title(f"{model_name} - Fold {fold}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            reports_dir = Path("reports/backtests")
            reports_dir.mkdir(parents=True, exist_ok=True)

            plot_path = reports_dir / f"{model_name}_fold_{fold}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info("Saved fold plot", path=str(plot_path))

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            self.logger.error(f"Failed to save fold plot: {e}")

    def _save_summary_plot(
        self,
        fold_results: list[dict],
        model_name: str,
    ) -> None:
        """Save summary plot across all folds.

        Args:
            fold_results: List of fold results.
            model_name: Name of the model.
        """
        try:
            import matplotlib.pyplot as plt

            # Create metrics comparison plot
            metrics_to_plot = ["RMSE", "MAPE", "sMAPE"]
            available_metrics = [m for m in metrics_to_plot if m in fold_results[0]["metrics"]]

            if not available_metrics:
                return

            fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 4))
            if len(available_metrics) == 1:
                axes = [axes]

            for i, metric in enumerate(available_metrics):
                fold_numbers = [fold["fold"] for fold in fold_results]
                metric_values = [fold["metrics"][metric] for fold in fold_results]

                axes[i].bar(fold_numbers, metric_values)
                axes[i].set_title(f"{metric} by Fold")
                axes[i].set_xlabel("Fold")
                axes[i].set_ylabel(metric)
                axes[i].grid(True, alpha=0.3)

            plt.suptitle(f"{model_name} - Backtest Results")
            plt.tight_layout()

            # Save plot
            reports_dir = Path("reports/backtests")
            reports_dir.mkdir(parents=True, exist_ok=True)

            plot_path = reports_dir / f"{model_name}_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info("Saved summary plot", path=str(plot_path))

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping summary plot")
        except Exception as e:
            self.logger.error(f"Failed to save summary plot: {e}")

    def _log_to_mlflow(
        self,
        fold_results: list[dict],
        aggregated_results: dict[str, float],
        model_name: str,
    ) -> None:
        """Log backtesting results to MLflow.

        Args:
            fold_results: List of fold results.
            aggregated_results: Aggregated metrics.
            model_name: Name of the model.
        """
        try:
            registry = ModelRegistry(self.settings)

            with registry.start_run(run_name=f"backtest_{model_name}"):
                # Log aggregated metrics
                registry.log_model_metrics(aggregated_results)

                # Log fold metrics
                for fold_result in fold_results:
                    fold_metrics = {f"fold_{fold_result['fold']}_{k}": v
                                  for k, v in fold_result["metrics"].items()}
                    registry.log_model_metrics(fold_metrics)

                # Log parameters
                params = {
                    "model_name": model_name,
                    "n_folds": len(fold_results),
                    "horizon": self.settings.default_horizon,
                }
                registry.log_model_parameters(params)

                self.logger.info("Logged backtest results to MLflow")

        except Exception as e:
            self.logger.warning(f"Failed to log to MLflow: {e}")


def rolling_backtest(
    df: pd.DataFrame,
    series_cols: list[str],
    target_col: str,
    date_col: str,
    horizon: int,
    n_folds: int,
    fit_fn: Callable,
    predict_fn: Callable,
    feature_fn: Callable,
    metrics: list[str],
    model_name: str = "model",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Convenience function for rolling backtesting.

    Args:
        df: Input DataFrame with time series data.
        series_cols: Columns that define series.
        target_col: Target variable column name.
        date_col: Date column name.
        horizon: Forecast horizon in days.
        n_folds: Number of backtesting folds.
        fit_fn: Function to fit model.
        predict_fn: Function to predict.
        feature_fn: Function to build features.
        metrics: List of metrics to calculate.
        model_name: Name of the model.
        settings: Application settings.

    Returns:
        Dictionary with backtesting results.
    """
    if settings is None:
        settings = Settings()

    backtester = RollingBacktester(settings)

    return backtester.rolling_backtest(
        df=df,
        series_cols=series_cols,
        target_col=target_col,
        date_col=date_col,
        horizon=horizon,
        n_folds=n_folds,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        feature_fn=feature_fn,
        metrics=metrics,
        model_name=model_name,
    )
