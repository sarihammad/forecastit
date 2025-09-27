"""Evaluation metrics for time series forecasting."""


import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class MetricsCalculator:
    """Calculate various evaluation metrics for time series forecasting."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = logger.bind(component="metrics_calculator")

    def calculate_metrics(
        self,
        y_true: np.ndarray | pd.Series | list,
        y_pred: np.ndarray | pd.Series | list,
        metrics: list[str] | None = None,
        y_train: np.ndarray | pd.Series | list | None = None,
    ) -> dict[str, float]:
        """Calculate multiple metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            metrics: List of metrics to calculate.
            y_train: Training values (for MASE calculation).

        Returns:
            Dictionary with metric values.
        """
        if metrics is None:
            metrics = ["RMSE", "MAE", "MAPE", "sMAPE", "R2"]

        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_train = np.asarray(y_train) if y_train is not None else None

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            self.logger.warning("No valid values for metric calculation")
            return {metric: np.nan for metric in metrics}

        results = {}

        for metric in metrics:
            try:
                if metric == "RMSE":
                    results[metric] = self._rmse(y_true, y_pred)
                elif metric == "MAE":
                    results[metric] = self._mae(y_true, y_pred)
                elif metric == "MAPE":
                    results[metric] = self._mape(y_true, y_pred)
                elif metric == "sMAPE":
                    results[metric] = self._smape(y_true, y_pred)
                elif metric == "MASE":
                    if y_train is not None:
                        results[metric] = self._mase(y_true, y_pred, y_train)
                    else:
                        results[metric] = np.nan
                elif metric == "R2":
                    results[metric] = self._r2(y_true, y_pred)
                elif metric == "Pinball":
                    results[metric] = self._pinball_loss(y_true, y_pred, alpha=0.5)
                else:
                    self.logger.warning(f"Unknown metric: {metric}")
                    results[metric] = np.nan
            except Exception as e:
                self.logger.error(f"Error calculating {metric}: {e}")
                results[metric] = np.nan

        return results

    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            RMSE value.
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def _mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            MAE value.
        """
        return np.mean(np.abs(y_true - y_pred))

    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            MAPE value.
        """
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return np.nan

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def _smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            sMAPE value.
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

        # Avoid division by zero
        mask = denominator != 0
        if not mask.any():
            return np.nan

        return np.mean(numerator[mask] / denominator[mask]) * 100

    def _mase(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonal_period: int = 1,
    ) -> float:
        """Calculate Mean Absolute Scaled Error.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            y_train: Training values.
            seasonal_period: Seasonal period for naive forecast.

        Returns:
            MASE value.
        """
        # Calculate naive forecast error
        if seasonal_period == 1:
            naive_errors = np.abs(np.diff(y_train))
        else:
            naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])

        if len(naive_errors) == 0:
            return np.nan

        naive_mae = np.mean(naive_errors)

        if naive_mae == 0:
            return np.nan

        # Calculate MASE
        forecast_mae = self._mae(y_true, y_pred)
        return forecast_mae / naive_mae

    def _r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            R-squared value.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return np.nan

        return 1 - (ss_res / ss_tot)

    def _pinball_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.5,
    ) -> float:
        """Calculate Pinball Loss for quantile regression.

        Args:
            y_true: True values.
            y_pred: Predicted values (quantile).
            alpha: Quantile level.

        Returns:
            Pinball loss value.
        """
        errors = y_true - y_pred
        return np.mean(np.maximum(alpha * errors, (alpha - 1) * errors))

    def calculate_quantile_metrics(
        self,
        y_true: np.ndarray,
        y_pred_lower: np.ndarray,
        y_pred_median: np.ndarray,
        y_pred_upper: np.ndarray,
    ) -> dict[str, float]:
        """Calculate metrics for quantile predictions.

        Args:
            y_true: True values.
            y_pred_lower: Lower quantile predictions.
            y_pred_median: Median predictions.
            y_pred_upper: Upper quantile predictions.

        Returns:
            Dictionary with quantile metrics.
        """
        results = {}

        # Point forecast metrics (using median)
        results.update(self.calculate_metrics(y_true, y_pred_median))

        # Quantile-specific metrics
        results["Pinball_0.1"] = self._pinball_loss(y_true, y_pred_lower, alpha=0.1)
        results["Pinball_0.5"] = self._pinball_loss(y_true, y_pred_median, alpha=0.5)
        results["Pinball_0.9"] = self._pinball_loss(y_true, y_pred_upper, alpha=0.9)

        # Coverage metrics
        results["Coverage_80"] = self._coverage(y_true, y_pred_lower, y_pred_upper, 0.8)
        results["Coverage_90"] = self._coverage(y_true, y_pred_lower, y_pred_upper, 0.9)

        # Interval width
        results["Interval_Width"] = np.mean(y_pred_upper - y_pred_lower)

        return results

    def _coverage(
        self,
        y_true: np.ndarray,
        y_pred_lower: np.ndarray,
        y_pred_upper: np.ndarray,
        confidence_level: float,
    ) -> float:
        """Calculate prediction interval coverage.

        Args:
            y_true: True values.
            y_pred_lower: Lower bound predictions.
            y_pred_upper: Upper bound predictions.
            confidence_level: Expected coverage level.

        Returns:
            Actual coverage percentage.
        """
        in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        return np.mean(in_interval) * 100

    def calculate_series_metrics(
        self,
        data: pd.DataFrame,
        y_true_column: str,
        y_pred_column: str,
        group_columns: list[str],
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Calculate metrics for each series.

        Args:
            data: DataFrame with predictions.
            y_true_column: Name of true values column.
            y_pred_column: Name of predicted values column.
            group_columns: Columns to group by.
            metrics: List of metrics to calculate.

        Returns:
            DataFrame with metrics per series.
        """
        if metrics is None:
            metrics = ["RMSE", "MAE", "MAPE", "sMAPE"]

        series_metrics = []

        for group, group_data in data.groupby(group_columns):
            y_true = group_data[y_true_column].values
            y_pred = group_data[y_pred_column].values

            metrics_dict = self.calculate_metrics(y_true, y_pred, metrics)

            # Add group information
            metrics_dict.update({
                col: group[i] for i, col in enumerate(group_columns)
            })

            series_metrics.append(metrics_dict)

        return pd.DataFrame(series_metrics)

    def aggregate_metrics(
        self,
        metrics_df: pd.DataFrame,
        metric_columns: list[str],
        aggregation_method: str = "mean",
    ) -> dict[str, float]:
        """Aggregate metrics across series.

        Args:
            metrics_df: DataFrame with metrics per series.
            metric_columns: List of metric columns to aggregate.
            aggregation_method: Method for aggregation ("mean", "median", "weighted").

        Returns:
            Dictionary with aggregated metrics.
        """
        aggregated = {}

        for metric in metric_columns:
            if metric not in metrics_df.columns:
                continue

            if aggregation_method == "mean":
                aggregated[metric] = metrics_df[metric].mean()
            elif aggregation_method == "median":
                aggregated[metric] = metrics_df[metric].median()
            elif aggregation_method == "weighted":
                # Weight by series length or volume (if available)
                weights = metrics_df.get("length", 1)
                aggregated[metric] = np.average(metrics_df[metric], weights=weights)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        return aggregated

    def compare_models(
        self,
        results_dict: dict[str, dict[str, float]],
        primary_metric: str = "RMSE",
    ) -> pd.DataFrame:
        """Compare multiple models.

        Args:
            results_dict: Dictionary with model results.
            primary_metric: Primary metric for ranking.

        Returns:
            DataFrame with model comparison.
        """
        comparison_data = []

        for model_name, metrics in results_dict.items():
            row = {"model": model_name}
            row.update(metrics)
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by primary metric
        if primary_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(primary_metric)

        return comparison_df
