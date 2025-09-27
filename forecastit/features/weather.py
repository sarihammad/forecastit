"""Weather feature engineering."""


import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class WeatherFeatures:
    """Generate weather-related features."""

    def __init__(self):
        """Initialize weather features."""
        self.logger = logger.bind(component="weather_features")

    def add_weather_features(
        self,
        data: pd.DataFrame,
        temperature_column: str = "temperature",
        precipitation_column: str = "precipitation",
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Add weather features to DataFrame.

        Args:
            data: DataFrame with weather data.
            temperature_column: Name of the temperature column.
            precipitation_column: Name of the precipitation column.
            group_columns: Columns to group by (e.g., ['store_id']).

        Returns:
            DataFrame with weather features added.
        """
        self.logger.info(
            "Adding weather features",
            shape=data.shape,
            temperature_column=temperature_column,
            precipitation_column=precipitation_column,
        )

        # Make a copy to avoid modifying original data
        result = data.copy()

        # Set default group columns
        if group_columns is None:
            group_columns = ["store_id"]

        # Sort by group columns and date
        sort_columns = [*group_columns, "date"]
        result = result.sort_values(sort_columns).reset_index(drop=True)

        # Add temperature features
        if temperature_column in result.columns:
            result = self._add_temperature_features(result, temperature_column, group_columns)

        # Add precipitation features
        if precipitation_column in result.columns:
            result = self._add_precipitation_features(result, precipitation_column, group_columns)

        # Add weather interaction features
        if temperature_column in result.columns and precipitation_column in result.columns:
            result = self._add_weather_interaction_features(
                result, temperature_column, precipitation_column
            )

        self.logger.info(
            "Weather features added",
            original_shape=data.shape,
            new_shape=result.shape,
            new_columns=set(result.columns) - set(data.columns),
        )

        return result

    def _add_temperature_features(
        self,
        data: pd.DataFrame,
        temperature_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Add temperature-related features.

        Args:
            data: DataFrame with temperature data.
            temperature_column: Name of the temperature column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with temperature features added.
        """
        # Temperature categories
        data["temp_category"] = pd.cut(
            data[temperature_column],
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=["very_cold", "cold", "mild", "warm", "hot"],
        )

        # Temperature extremes
        data["is_very_cold"] = data[temperature_column] < 0
        data["is_cold"] = (data[temperature_column] >= 0) & (data[temperature_column] < 10)
        data["is_mild"] = (data[temperature_column] >= 10) & (data[temperature_column] < 20)
        data["is_warm"] = (data[temperature_column] >= 20) & (data[temperature_column] < 30)
        data["is_hot"] = data[temperature_column] >= 30

        # Temperature change
        data["temp_change"] = data.groupby(group_columns)[temperature_column].diff()
        data["temp_change_abs"] = data["temp_change"].abs()

        # Rolling temperature statistics
        for window in [3, 7, 14]:
            # Rolling mean temperature
            temp_mean_column = f"temp_rolling_mean_{window}"
            data[temp_mean_column] = (
                data.groupby(group_columns)[temperature_column]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

            # Rolling temperature range
            temp_range_column = f"temp_rolling_range_{window}"
            data[temp_range_column] = (
                data.groupby(group_columns)[temperature_column]
                .rolling(window=window, min_periods=1)
                .apply(lambda x: x.max() - x.min())
                .reset_index(0, drop=True)
            )

            # Rolling temperature volatility
            temp_std_column = f"temp_rolling_std_{window}"
            data[temp_std_column] = (
                data.groupby(group_columns)[temperature_column]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )

        # Temperature lag features
        for lag in [1, 3, 7]:
            temp_lag_column = f"temp_lag_{lag}"
            data[temp_lag_column] = data.groupby(group_columns)[temperature_column].shift(lag)

        # Temperature trend
        data["temp_trend"] = (
            data.groupby(group_columns)[temperature_column]
            .rolling(window=7, min_periods=1)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            .reset_index(0, drop=True)
        )

        # Seasonal temperature deviation
        data["temp_seasonal_deviation"] = (
            data.groupby(group_columns)[temperature_column]
            .rolling(window=365, min_periods=1)
            .apply(lambda x: x.iloc[-1] - x.mean())
            .reset_index(0, drop=True)
        )

        return data

    def _add_precipitation_features(
        self,
        data: pd.DataFrame,
        precipitation_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Add precipitation-related features.

        Args:
            data: DataFrame with precipitation data.
            precipitation_column: Name of the precipitation column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with precipitation features added.
        """
        # Precipitation categories
        data["precip_category"] = pd.cut(
            data[precipitation_column],
            bins=[0, 0.1, 2.5, 10, 25, np.inf],
            labels=["none", "light", "moderate", "heavy", "extreme"],
        )

        # Precipitation flags
        data["is_dry"] = data[precipitation_column] == 0
        data["is_light_rain"] = (data[precipitation_column] > 0) & (data[precipitation_column] <= 2.5)
        data["is_moderate_rain"] = (data[precipitation_column] > 2.5) & (data[precipitation_column] <= 10)
        data["is_heavy_rain"] = (data[precipitation_column] > 10) & (data[precipitation_column] <= 25)
        data["is_extreme_rain"] = data[precipitation_column] > 25

        # Precipitation change
        data["precip_change"] = data.groupby(group_columns)[precipitation_column].diff()

        # Rolling precipitation statistics
        for window in [3, 7, 14]:
            # Rolling mean precipitation
            precip_mean_column = f"precip_rolling_mean_{window}"
            data[precip_mean_column] = (
                data.groupby(group_columns)[precipitation_column]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

            # Rolling total precipitation
            precip_total_column = f"precip_rolling_total_{window}"
            data[precip_total_column] = (
                data.groupby(group_columns)[precipitation_column]
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(0, drop=True)
            )

            # Rolling maximum precipitation
            precip_max_column = f"precip_rolling_max_{window}"
            data[precip_max_column] = (
                data.groupby(group_columns)[precipitation_column]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(0, drop=True)
            )

        # Precipitation lag features
        for lag in [1, 3, 7]:
            precip_lag_column = f"precip_lag_{lag}"
            data[precip_lag_column] = data.groupby(group_columns)[precipitation_column].shift(lag)

        # Dry spell duration
        data["dry_spell_duration"] = (
            data.groupby(group_columns)[precipitation_column]
            .apply(lambda x: x.groupby((x > 0).cumsum()).cumcount())
            .reset_index(0, drop=True)
        )

        # Wet spell duration
        data["wet_spell_duration"] = (
            data.groupby(group_columns)[precipitation_column]
            .apply(lambda x: x.groupby((x == 0).cumsum()).cumcount())
            .reset_index(0, drop=True)
        )

        return data

    def _add_weather_interaction_features(
        self,
        data: pd.DataFrame,
        temperature_column: str,
        precipitation_column: str,
    ) -> pd.DataFrame:
        """Add weather interaction features.

        Args:
            data: DataFrame with temperature and precipitation data.
            temperature_column: Name of the temperature column.
            precipitation_column: Name of the precipitation column.

        Returns:
            DataFrame with weather interaction features added.
        """
        # Weather comfort index (simplified)
        data["weather_comfort"] = (
            data[temperature_column] * (1 - data[precipitation_column] / 100)
        )

        # Bad weather flag (cold and wet)
        data["is_bad_weather"] = (
            (data[temperature_column] < 10) & (data[precipitation_column] > 5)
        )

        # Good weather flag (mild and dry)
        data["is_good_weather"] = (
            (data[temperature_column] >= 15) & (data[temperature_column] <= 25) &
            (data[precipitation_column] < 1)
        )

        # Extreme weather flag
        data["is_extreme_weather"] = (
            (data[temperature_column] < -5) | (data[temperature_column] > 35) |
            (data[precipitation_column] > 25)
        )

        # Weather severity score
        data["weather_severity"] = (
            abs(data[temperature_column] - 20) / 20 +  # Temperature deviation from ideal
            data[precipitation_column] / 25  # Precipitation intensity
        )

        return data

    def get_temperature_feature_names(self, temperature_column: str = "temperature") -> list[str]:
        """Get list of temperature feature names.

        Args:
            temperature_column: Name of the temperature column.

        Returns:
            List of temperature feature names.
        """
        features = [
            "temp_category",
            "is_very_cold", "is_cold", "is_mild", "is_warm", "is_hot",
            "temp_change", "temp_change_abs",
            "temp_trend",
            "temp_seasonal_deviation",
        ]

        # Rolling temperature features
        for window in [3, 7, 14]:
            features.extend([
                f"temp_rolling_mean_{window}",
                f"temp_rolling_range_{window}",
                f"temp_rolling_std_{window}",
            ])

        # Temperature lag features
        for lag in [1, 3, 7]:
            features.append(f"temp_lag_{lag}")

        return features

    def get_precipitation_feature_names(self, precipitation_column: str = "precipitation") -> list[str]:
        """Get list of precipitation feature names.

        Args:
            precipitation_column: Name of the precipitation column.

        Returns:
            List of precipitation feature names.
        """
        features = [
            "precip_category",
            "is_dry", "is_light_rain", "is_moderate_rain", "is_heavy_rain", "is_extreme_rain",
            "precip_change",
            "dry_spell_duration", "wet_spell_duration",
        ]

        # Rolling precipitation features
        for window in [3, 7, 14]:
            features.extend([
                f"precip_rolling_mean_{window}",
                f"precip_rolling_total_{window}",
                f"precip_rolling_max_{window}",
            ])

        # Precipitation lag features
        for lag in [1, 3, 7]:
            features.append(f"precip_lag_{lag}")

        return features

    def get_weather_interaction_feature_names(self) -> list[str]:
        """Get list of weather interaction feature names.

        Returns:
            List of weather interaction feature names.
        """
        return [
            "weather_comfort",
            "is_bad_weather",
            "is_good_weather",
            "is_extreme_weather",
            "weather_severity",
        ]
