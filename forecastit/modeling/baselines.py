"""Baseline forecasting models."""


import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class BaselineModels:
    """Simple baseline forecasting models."""

    def __init__(self):
        """Initialize baseline models."""
        self.logger = logger.bind(component="baseline_models")

    def naive_forecast(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        horizon: int = 28,
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate naive forecasts (last value).

        Args:
            data: Historical data.
            target_column: Name of target column.
            horizon: Forecast horizon.
            group_columns: Columns to group by.

        Returns:
            DataFrame with naive forecasts.
        """
        if group_columns is None:
            group_columns = ["store_id", "item_id"]

        self.logger.info(
            "Generating naive forecasts",
            horizon=horizon,
            group_columns=group_columns,
        )

        forecasts = []

        for group, group_data in data.groupby(group_columns):
            # Get last observed value
            last_value = group_data[target_column].iloc[-1]

            # Generate forecast dates
            last_date = group_data["date"].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq="D"
            )

            # Create forecast DataFrame
            for date in forecast_dates:
                forecast_row = {col: group[i] for i, col in enumerate(group_columns)}
                forecast_row.update({
                    "date": date,
                    target_column: last_value,
                    "forecast_type": "naive",
                })
                forecasts.append(forecast_row)

        return pd.DataFrame(forecasts)

    def seasonal_naive_forecast(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        horizon: int = 28,
        seasonal_period: int = 7,
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate seasonal naive forecasts.

        Args:
            data: Historical data.
            target_column: Name of target column.
            horizon: Forecast horizon.
            seasonal_period: Seasonal period (e.g., 7 for weekly).
            group_columns: Columns to group by.

        Returns:
            DataFrame with seasonal naive forecasts.
        """
        if group_columns is None:
            group_columns = ["store_id", "item_id"]

        self.logger.info(
            "Generating seasonal naive forecasts",
            horizon=horizon,
            seasonal_period=seasonal_period,
            group_columns=group_columns,
        )

        forecasts = []

        for group, group_data in data.groupby(group_columns):
            # Get seasonal values
            if len(group_data) < seasonal_period:
                # Fallback to naive if not enough data
                last_value = group_data[target_column].iloc[-1]
                seasonal_values = [last_value] * seasonal_period
            else:
                seasonal_values = group_data[target_column].tail(seasonal_period).tolist()

            # Generate forecast dates
            last_date = group_data["date"].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq="D"
            )

            # Create forecast DataFrame
            for i, date in enumerate(forecast_dates):
                # Use seasonal pattern
                seasonal_index = i % seasonal_period
                forecast_value = seasonal_values[seasonal_index]

                forecast_row = {col: group[j] for j, col in enumerate(group_columns)}
                forecast_row.update({
                    "date": date,
                    target_column: forecast_value,
                    "forecast_type": "seasonal_naive",
                })
                forecasts.append(forecast_row)

        return pd.DataFrame(forecasts)

    def moving_average_forecast(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        horizon: int = 28,
        window: int = 7,
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate moving average forecasts.

        Args:
            data: Historical data.
            target_column: Name of target column.
            horizon: Forecast horizon.
            window: Moving average window.
            group_columns: Columns to group by.

        Returns:
            DataFrame with moving average forecasts.
        """
        if group_columns is None:
            group_columns = ["store_id", "item_id"]

        self.logger.info(
            "Generating moving average forecasts",
            horizon=horizon,
            window=window,
            group_columns=group_columns,
        )

        forecasts = []

        for group, group_data in data.groupby(group_columns):
            # Calculate moving average
            if len(group_data) < window:
                # Fallback to naive if not enough data
                forecast_value = group_data[target_column].mean()
            else:
                forecast_value = group_data[target_column].tail(window).mean()

            # Generate forecast dates
            last_date = group_data["date"].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq="D"
            )

            # Create forecast DataFrame
            for date in forecast_dates:
                forecast_row = {col: group[i] for i, col in enumerate(group_columns)}
                forecast_row.update({
                    "date": date,
                    target_column: forecast_value,
                    "forecast_type": "moving_average",
                })
                forecasts.append(forecast_row)

        return pd.DataFrame(forecasts)

    def exponential_smoothing_forecast(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        horizon: int = 28,
        alpha: float = 0.3,
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate exponential smoothing forecasts.

        Args:
            data: Historical data.
            target_column: Name of target column.
            horizon: Forecast horizon.
            alpha: Smoothing parameter.
            group_columns: Columns to group by.

        Returns:
            DataFrame with exponential smoothing forecasts.
        """
        if group_columns is None:
            group_columns = ["store_id", "item_id"]

        self.logger.info(
            "Generating exponential smoothing forecasts",
            horizon=horizon,
            alpha=alpha,
            group_columns=group_columns,
        )

        forecasts = []

        for group, group_data in data.groupby(group_columns):
            # Calculate exponential smoothing
            values = group_data[target_column].values
            if len(values) == 0:
                forecast_value = 0
            else:
                # Simple exponential smoothing
                smoothed = np.zeros_like(values, dtype=float)
                smoothed[0] = values[0]

                for i in range(1, len(values)):
                    smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]

                forecast_value = smoothed[-1]

            # Generate forecast dates
            last_date = group_data["date"].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq="D"
            )

            # Create forecast DataFrame
            for date in forecast_dates:
                forecast_row = {col: group[i] for i, col in enumerate(group_columns)}
                forecast_row.update({
                    "date": date,
                    target_column: forecast_value,
                    "forecast_type": "exponential_smoothing",
                })
                forecasts.append(forecast_row)

        return pd.DataFrame(forecasts)

    def generate_all_baselines(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        horizon: int = 28,
        group_columns: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Generate all baseline forecasts.

        Args:
            data: Historical data.
            target_column: Name of target column.
            horizon: Forecast horizon.
            group_columns: Columns to group by.

        Returns:
            Dictionary with all baseline forecasts.
        """
        self.logger.info("Generating all baseline forecasts", horizon=horizon)

        baselines = {}

        # Naive forecast
        baselines["naive"] = self.naive_forecast(
            data, target_column, horizon, group_columns
        )

        # Seasonal naive forecast
        baselines["seasonal_naive"] = self.seasonal_naive_forecast(
            data, target_column, horizon, seasonal_period=7, group_columns=group_columns
        )

        # Moving average forecast
        baselines["moving_average"] = self.moving_average_forecast(
            data, target_column, horizon, window=7, group_columns=group_columns
        )

        # Exponential smoothing forecast
        baselines["exponential_smoothing"] = self.exponential_smoothing_forecast(
            data, target_column, horizon, alpha=0.3, group_columns=group_columns
        )

        self.logger.info(
            "Generated baseline forecasts",
            models=list(baselines.keys()),
            total_forecasts=sum(len(df) for df in baselines.values()),
        )

        return baselines
