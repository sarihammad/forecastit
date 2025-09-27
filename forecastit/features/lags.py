"""Lag and rolling window feature engineering."""


import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class LagFeatures:
    """Generate lag and rolling window features."""

    def __init__(self, max_lag: int = 28, rolling_windows: list[int] | None = None):
        """Initialize lag features.

        Args:
            max_lag: Maximum lag to create.
            rolling_windows: List of rolling window sizes.
        """
        if rolling_windows is None:
            rolling_windows = [7, 14, 28]

        self.max_lag = max_lag
        self.rolling_windows = rolling_windows
        self.logger = logger.bind(component="lag_features")

    def add_lag_features(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Add lag features to DataFrame.

        Args:
            data: DataFrame with time series data.
            target_column: Name of the target column.
            group_columns: Columns to group by (e.g., ['store_id', 'item_id']).

        Returns:
            DataFrame with lag features added.
        """
        self.logger.info(
            "Adding lag features",
            shape=data.shape,
            target_column=target_column,
            max_lag=self.max_lag,
            rolling_windows=self.rolling_windows,
        )

        # Make a copy to avoid modifying original data
        result = data.copy()

        # Set default group columns
        if group_columns is None:
            group_columns = ["store_id", "item_id"]

        # Sort by group columns and date
        sort_columns = [*group_columns, "date"]
        result = result.sort_values(sort_columns).reset_index(drop=True)

        # Add lag features
        result = self._add_lag_features(result, target_column, group_columns)

        # Add rolling features
        result = self._add_rolling_features(result, target_column, group_columns)

        # Add exponential moving average features
        result = self._add_ema_features(result, target_column, group_columns)

        # Add difference features
        result = self._add_difference_features(result, target_column, group_columns)

        self.logger.info(
            "Lag features added",
            original_shape=data.shape,
            new_shape=result.shape,
            new_columns=set(result.columns) - set(data.columns),
        )

        return result

    def _add_lag_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Add lag features.

        Args:
            data: DataFrame with time series data.
            target_column: Name of the target column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with lag features added.
        """
        # Create lag features for different periods
        lag_periods = [1, 7, 14, 28]

        for lag in lag_periods:
            if lag <= self.max_lag:
                lag_column = f"{target_column}_lag_{lag}"
                data[lag_column] = data.groupby(group_columns)[target_column].shift(lag)

        return data

    def _add_rolling_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Add rolling window features.

        Args:
            data: DataFrame with time series data.
            target_column: Name of the target column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with rolling features added.
        """
        for window in self.rolling_windows:
            if window <= self.max_lag:
                # Rolling mean
                mean_column = f"{target_column}_rolling_mean_{window}"
                rolling_mean = (
                    data.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                data[mean_column] = rolling_mean.values

                # Rolling standard deviation
                std_column = f"{target_column}_rolling_std_{window}"
                rolling_std = (
                    data.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )
                data[std_column] = rolling_std.values

                # Rolling minimum
                min_column = f"{target_column}_rolling_min_{window}"
                rolling_min = (
                    data.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .min()
                    .reset_index(0, drop=True)
                )
                data[min_column] = rolling_min.values

                # Rolling maximum
                max_column = f"{target_column}_rolling_max_{window}"
                rolling_max = (
                    data.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .max()
                    .reset_index(0, drop=True)
                )
                data[max_column] = rolling_max.values

                # Rolling median
                median_column = f"{target_column}_rolling_median_{window}"
                rolling_median = (
                    data.groupby(group_columns)[target_column]
                    .rolling(window=window, min_periods=1)
                    .median()
                    .reset_index(0, drop=True)
                )
                data[median_column] = rolling_median.values

        return data

    def _add_ema_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        group_columns: list[str],
        alpha: float = 0.3,
    ) -> pd.DataFrame:
        """Add exponential moving average features.

        Args:
            data: DataFrame with time series data.
            target_column: Name of the target column.
            group_columns: Columns to group by.
            alpha: EMA smoothing factor.

        Returns:
            DataFrame with EMA features added.
        """
        ema_column = f"{target_column}_ema_{int(1/alpha)}"
        ema_values = (
            data.groupby(group_columns)[target_column]
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .reset_index(0, drop=True)
        )
        data[ema_column] = ema_values.values

        return data

    def _add_difference_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Add difference features.

        Args:
            data: DataFrame with time series data.
            target_column: Name of the target column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with difference features added.
        """
        # First difference
        diff_column = f"{target_column}_diff_1"
        data[diff_column] = data.groupby(group_columns)[target_column].diff(1)

        # Second difference
        diff2_column = f"{target_column}_diff_2"
        data[diff2_column] = data.groupby(group_columns)[target_column].diff(2)

        # Weekly difference
        diff7_column = f"{target_column}_diff_7"
        data[diff7_column] = data.groupby(group_columns)[target_column].diff(7)

        return data

    def add_promo_lag_features(
        self,
        data: pd.DataFrame,
        promo_column: str = "on_promo",
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Add promotion-related lag features.

        Args:
            data: DataFrame with promotion data.
            promo_column: Name of the promotion column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with promo lag features added.
        """
        self.logger.info(
            "Adding promo lag features",
            promo_column=promo_column,
            rolling_windows=self.rolling_windows,
        )

        # Set default group columns
        if group_columns is None:
            group_columns = ["store_id", "item_id"]

        # Sort by group columns and date
        sort_columns = [*group_columns, "date"]
        data = data.sort_values(sort_columns).reset_index(drop=True)

        # Rolling promo count
        for window in self.rolling_windows:
            promo_count_column = f"{promo_column}_rolling_{window}"
            data[promo_count_column] = (
                data.groupby(group_columns)[promo_column]
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(0, drop=True)
            )

        # Days since last promotion
        data["days_since_last_promo"] = (
            data.groupby(group_columns)[promo_column]
            .apply(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount())
            .reset_index(0, drop=True)
        )

        # Days until next promotion
        data["days_until_next_promo"] = (
            data.groupby(group_columns)[promo_column]
            .apply(lambda x: x[::-1].groupby((x[::-1] != x[::-1].shift()).cumsum()).cumcount()[::-1])
            .reset_index(0, drop=True)
        )

        return data

    def get_lag_feature_names(self, target_column: str = "sales") -> list[str]:
        """Get list of lag feature names.

        Args:
            target_column: Name of the target column.

        Returns:
            List of lag feature names.
        """
        features = []

        # Lag features
        lag_periods = [1, 7, 14, 28]
        for lag in lag_periods:
            if lag <= self.max_lag:
                features.append(f"{target_column}_lag_{lag}")

        # Rolling features
        for window in self.rolling_windows:
            if window <= self.max_lag:
                features.extend([
                    f"{target_column}_rolling_mean_{window}",
                    f"{target_column}_rolling_std_{window}",
                    f"{target_column}_rolling_min_{window}",
                    f"{target_column}_rolling_max_{window}",
                    f"{target_column}_rolling_median_{window}",
                ])

        # EMA features
        features.extend([
            f"{target_column}_ema_3",  # alpha=0.3
        ])

        # Difference features
        features.extend([
            f"{target_column}_diff_1",
            f"{target_column}_diff_2",
            f"{target_column}_diff_7",
        ])

        return features

    def get_promo_lag_feature_names(self, promo_column: str = "on_promo") -> list[str]:
        """Get list of promo lag feature names.

        Args:
            promo_column: Name of the promotion column.

        Returns:
            List of promo lag feature names.
        """
        features = []

        # Rolling promo features
        for window in self.rolling_windows:
            features.append(f"{promo_column}_rolling_{window}")

        # Days since/until promo
        features.extend([
            "days_since_last_promo",
            "days_until_next_promo",
        ])

        return features
