"""Price and promotion feature engineering."""


import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class PricePromoFeatures:
    """Generate price and promotion-related features."""

    def __init__(self):
        """Initialize price and promotion features."""
        self.logger = logger.bind(component="price_promo_features")

    def add_price_promo_features(
        self,
        data: pd.DataFrame,
        price_column: str = "price",
        promo_column: str = "on_promo",
        sales_column: str = "sales",
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Add price and promotion features to DataFrame.

        Args:
            data: DataFrame with price and promotion data.
            price_column: Name of the price column.
            promo_column: Name of the promotion column.
            sales_column: Name of the sales column.
            group_columns: Columns to group by (e.g., ['store_id', 'item_id']).

        Returns:
            DataFrame with price and promotion features added.
        """
        self.logger.info(
            "Adding price and promotion features",
            shape=data.shape,
            price_column=price_column,
            promo_column=promo_column,
            sales_column=sales_column,
        )

        # Make a copy to avoid modifying original data
        result = data.copy()

        # Set default group columns
        if group_columns is None:
            group_columns = ["store_id", "item_id"]

        # Sort by group columns and date
        sort_columns = [*group_columns, "date"]
        result = result.sort_values(sort_columns).reset_index(drop=True)

        # Add price features
        result = self._add_price_features(result, price_column, group_columns)

        # Add promotion features
        result = self._add_promotion_features(result, promo_column, group_columns)

        # Add price elasticity features
        result = self._add_price_elasticity_features(
            result, price_column, sales_column, group_columns
        )

        # Add promotion effectiveness features
        result = self._add_promotion_effectiveness_features(
            result, promo_column, sales_column, group_columns
        )

        self.logger.info(
            "Price and promotion features added",
            original_shape=data.shape,
            new_shape=result.shape,
            new_columns=set(result.columns) - set(data.columns),
        )

        return result

    def _add_price_features(
        self,
        data: pd.DataFrame,
        price_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Add price-related features.

        Args:
            data: DataFrame with price data.
            price_column: Name of the price column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with price features added.
        """
        # Price percentage change
        price_pct_change_column = f"{price_column}_pct_change"
        data[price_pct_change_column] = (
            data.groupby(group_columns)[price_column].pct_change()
        )

        # Price difference
        price_diff_column = f"{price_column}_diff"
        data[price_diff_column] = data.groupby(group_columns)[price_column].diff()

        # Rolling price statistics
        for window in [7, 14, 28]:
            # Rolling mean price
            price_mean_column = f"{price_column}_rolling_mean_{window}"
            rolling_mean = (
                data.groupby(group_columns)[price_column]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            data[price_mean_column] = rolling_mean.values

            # Rolling standard deviation of price
            price_std_column = f"{price_column}_rolling_std_{window}"
            rolling_std = (
                data.groupby(group_columns)[price_column]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
            data[price_std_column] = rolling_std.values

            # Price volatility
            price_volatility_column = f"{price_column}_volatility_{window}"
            rolling_std = (
                data.groupby(group_columns)[price_column]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
            rolling_mean = (
                data.groupby(group_columns)[price_column]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            data[price_volatility_column] = rolling_std.values / rolling_mean.values

        # Price position relative to rolling statistics
        for window in [7, 14, 28]:
            # Price percentile
            price_percentile_column = f"{price_column}_percentile_{window}"
            rolling_percentile = (
                data.groupby(group_columns)[price_column]
                .rolling(window=window, min_periods=1)
                .apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5)
                .reset_index(0, drop=True)
            )
            data[price_percentile_column] = rolling_percentile.values

        # Price trend
        def safe_polyfit(x):
            try:
                if len(x) > 1:
                    return np.polyfit(range(len(x)), x, 1)[0]
                else:
                    return 0.0
            except (np.linalg.LinAlgError, ValueError):
                return 0.0
        
        rolling_trend = (
            data.groupby(group_columns)[price_column]
            .rolling(window=7, min_periods=1)
            .apply(safe_polyfit)
            .reset_index(0, drop=True)
        )
        data["price_trend"] = rolling_trend.values

        return data

    def _add_promotion_features(
        self,
        data: pd.DataFrame,
        promo_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Add promotion-related features.

        Args:
            data: DataFrame with promotion data.
            promo_column: Name of the promotion column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with promotion features added.
        """
        # Rolling promotion count
        for window in [7, 14, 28]:
            promo_count_column = f"{promo_column}_rolling_{window}"
            rolling_sum = (
                data.groupby(group_columns)[promo_column]
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(0, drop=True)
            )
            data[promo_count_column] = rolling_sum.values

            # Promotion frequency
            promo_freq_column = f"{promo_column}_frequency_{window}"
            rolling_mean = (
                data.groupby(group_columns)[promo_column]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            data[promo_freq_column] = rolling_mean.values

        # Days since last promotion
        days_since = (
            data.groupby(group_columns)[promo_column]
            .apply(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount())
            .reset_index(0, drop=True)
        )
        data["days_since_last_promo"] = days_since.values

        # Days until next promotion
        days_until = (
            data.groupby(group_columns)[promo_column]
            .apply(lambda x: x[::-1].groupby((x[::-1] != x[::-1].shift()).cumsum()).cumcount()[::-1])
            .reset_index(0, drop=True)
        )
        data["days_until_next_promo"] = days_until.values

        # Promotion duration (consecutive days)
        promo_duration = (
            data.groupby(group_columns)[promo_column]
            .apply(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount())
            .reset_index(0, drop=True)
        )
        data["promo_duration"] = promo_duration.values

        # Promotion intensity (rolling sum of consecutive promotions)
        promo_intensity = (
            data.groupby(group_columns)[promo_column]
            .apply(lambda x: x.groupby((x != x.shift()).cumsum()).cumsum())
            .reset_index(0, drop=True)
        )
        data["promo_intensity"] = promo_intensity.values

        return data

    def _add_price_elasticity_features(
        self,
        data: pd.DataFrame,
        price_column: str,
        sales_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Add price elasticity features.

        Args:
            data: DataFrame with price and sales data.
            price_column: Name of the price column.
            sales_column: Name of the sales column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with price elasticity features added.
        """
        # Price elasticity (percentage change in sales / percentage change in price)
        price_elasticity_column = "price_elasticity"

        # Calculate percentage changes
        price_pct_change = data.groupby(group_columns)[price_column].pct_change()
        sales_pct_change = data.groupby(group_columns)[sales_column].pct_change()

        # Calculate elasticity (avoid division by zero)
        elasticity = sales_pct_change / price_pct_change
        elasticity = elasticity.replace([np.inf, -np.inf], np.nan)

        data[price_elasticity_column] = elasticity

        # Rolling price elasticity
        for window in [14, 28]:
            elasticity_rolling_column = f"price_elasticity_rolling_{window}"
            rolling_elasticity = (
                data.groupby(group_columns)[price_elasticity_column]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            data[elasticity_rolling_column] = rolling_elasticity.values

        # Price sensitivity (standard deviation of elasticity)
        rolling_sensitivity = (
            data.groupby(group_columns)[price_elasticity_column]
            .rolling(window=28, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        data["price_sensitivity"] = rolling_sensitivity.values

        return data

    def _add_promotion_effectiveness_features(
        self,
        data: pd.DataFrame,
        promo_column: str,
        sales_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Add promotion effectiveness features.

        Args:
            data: DataFrame with promotion and sales data.
            promo_column: Name of the promotion column.
            sales_column: Name of the sales column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with promotion effectiveness features added.
        """
        # Sales lift during promotions
        sales_lift_column = "promo_sales_lift"

        # Calculate rolling mean sales for non-promo days
        non_promo_sales = data[data[promo_column] == 0].copy()
        if len(non_promo_sales) > 0:
            # Calculate baseline sales using rolling mean
            data["baseline_sales"] = (
                data.groupby(group_columns)[sales_column]
                .rolling(window=14, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
                .values
            )

            # Calculate sales lift

            data[sales_lift_column] = (
                (data[sales_column] - data["baseline_sales"]) / data["baseline_sales"]
            ).replace([np.inf, -np.inf], np.nan)

        # Promotion effectiveness (rolling mean of sales lift)
        for window in [14, 28]:
            effectiveness_column = f"promo_effectiveness_{window}"
            rolling_effectiveness = (
                data.groupby(group_columns)[sales_lift_column]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            data[effectiveness_column] = rolling_effectiveness.values

        # Promotion cannibalization (negative sales after promotion)
        promo_cannibalization = (
            data.groupby(group_columns)[sales_lift_column]
            .shift(-1)  # Look at next day's sales lift
            .where(data[promo_column] == 1, 0)  # Only for promotion days
        )
        data["promo_cannibalization"] = promo_cannibalization.values

        return data

    def get_price_feature_names(self, price_column: str = "price") -> list[str]:
        """Get list of price feature names.

        Args:
            price_column: Name of the price column.

        Returns:
            List of price feature names.
        """
        features = []

        # Basic price features
        features.extend([
            f"{price_column}_pct_change",
            f"{price_column}_diff",
        ])

        # Rolling price features
        for window in [7, 14, 28]:
            features.extend([
                f"{price_column}_rolling_mean_{window}",
                f"{price_column}_rolling_std_{window}",
                f"{price_column}_volatility_{window}",
                f"{price_column}_percentile_{window}",
            ])

        # Price trend
        features.append("price_trend")

        return features

    def get_promo_feature_names(self, promo_column: str = "on_promo") -> list[str]:
        """Get list of promotion feature names.

        Args:
            promo_column: Name of the promotion column.

        Returns:
            List of promotion feature names.
        """
        features = []

        # Rolling promotion features
        for window in [7, 14, 28]:
            features.extend([
                f"{promo_column}_rolling_{window}",
                f"{promo_column}_frequency_{window}",
            ])

        # Promotion timing features
        features.extend([
            "days_since_last_promo",
            "days_until_next_promo",
            "promo_duration",
            "promo_intensity",
        ])

        return features

    def get_elasticity_feature_names(self) -> list[str]:
        """Get list of price elasticity feature names.

        Returns:
            List of elasticity feature names.
        """
        features = [
            "price_elasticity",
            "price_sensitivity",
        ]

        # Rolling elasticity features
        for window in [14, 28]:
            features.append(f"price_elasticity_rolling_{window}")

        return features

    def get_effectiveness_feature_names(self) -> list[str]:
        """Get list of promotion effectiveness feature names.

        Returns:
            List of effectiveness feature names.
        """
        features = [
            "promo_sales_lift",
            "promo_cannibalization",
        ]

        # Rolling effectiveness features
        for window in [14, 28]:
            features.append(f"promo_effectiveness_{window}")

        return features
