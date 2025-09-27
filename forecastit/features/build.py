"""Feature building orchestration."""


import pandas as pd
import structlog

from forecastit.config.settings import Settings
from forecastit.features.calendar import CalendarFeatures
from forecastit.features.lags import LagFeatures
from forecastit.features.price_promo import PricePromoFeatures
from forecastit.features.weather import WeatherFeatures

logger = structlog.get_logger(__name__)


class FeatureBuilder:
    """Orchestrate feature engineering pipeline."""

    def __init__(self, settings: Settings):
        """Initialize feature builder.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.logger = logger.bind(component="feature_builder")

        # Initialize feature builders
        self.calendar_features = CalendarFeatures()
        self.lag_features = LagFeatures(
            max_lag=settings.max_lag,
            rolling_windows=settings.rolling_windows,
        )
        self.price_promo_features = PricePromoFeatures()
        self.weather_features = WeatherFeatures()

        # Fit/transform state
        self._is_fitted = False
        self._feature_columns = []
        self._categorical_encoders = {}
        self._scalers = {}

    def build_features(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        include_calendar: bool = True,
        include_lags: bool = True,
        include_price_promo: bool = True,
        include_weather: bool = True,
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build all features for the dataset.

        Args:
            data: Cleaned DataFrame.
            target_column: Name of the target column.
            include_calendar: Whether to include calendar features.
            include_lags: Whether to include lag features.
            include_price_promo: Whether to include price/promo features.
            include_weather: Whether to include weather features.
            group_columns: Columns to group by for time series features.

        Returns:
            DataFrame with all features added.
        """
        self.logger.info(
            "Building features",
            shape=data.shape,
            target_column=target_column,
            include_calendar=include_calendar,
            include_lags=include_lags,
            include_price_promo=include_price_promo,
            include_weather=include_weather,
        )

        # Make a copy to avoid modifying original data
        result = data.copy()

        # Set default group columns
        if group_columns is None:
            group_columns = ["store_id", "item_id"]

        # Build features in order (calendar first, then lags, then others)
        if include_calendar:
            result = self._build_calendar_features(result)

        if include_lags:
            result = self._build_lag_features(result, target_column, group_columns)

        if include_price_promo:
            result = self._build_price_promo_features(result, group_columns)

        if include_weather:
            result = self._build_weather_features(result, group_columns)

        # Add feature engineering flags
        result = self._add_feature_flags(result, data)

        # Validate features
        self._validate_features(result, data)

        self.logger.info(
            "Feature building completed",
            original_shape=data.shape,
            final_shape=result.shape,
            features_added=len(result.columns) - len(data.columns),
        )

        return result

    def _build_calendar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build calendar features.

        Args:
            data: DataFrame to add calendar features to.

        Returns:
            DataFrame with calendar features added.
        """
        self.logger.info("Building calendar features")

        # Only include calendar features if enabled in settings
        if not self.settings.include_calendar_features:
            self.logger.info("Calendar features disabled in settings")
            return data

        return self.calendar_features.add_calendar_features(data)

    def _build_lag_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Build lag and rolling features.

        Args:
            data: DataFrame to add lag features to.
            target_column: Name of the target column.
            group_columns: Columns to group by.

        Returns:
            DataFrame with lag features added.
        """
        self.logger.info("Building lag features")

        # Only include lag features if enabled in settings
        if not self.settings.include_lag_features:
            self.logger.info("Lag features disabled in settings")
            return data

        # Add lag features for target variable
        data = self.lag_features.add_lag_features(
            data, target_column=target_column, group_columns=group_columns
        )

        # Add promo lag features if promo column exists
        if "on_promo" in data.columns:
            data = self.lag_features.add_promo_lag_features(
                data, promo_column="on_promo", group_columns=group_columns
            )

        return data

    def _build_price_promo_features(
        self,
        data: pd.DataFrame,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Build price and promotion features.

        Args:
            data: DataFrame to add price/promo features to.
            group_columns: Columns to group by.

        Returns:
            DataFrame with price/promo features added.
        """
        self.logger.info("Building price and promotion features")

        # Only include price/promo features if enabled in settings
        if not self.settings.include_price_promo_features:
            self.logger.info("Price/promo features disabled in settings")
            return data

        # Check if required columns exist
        has_price = "price" in data.columns
        has_promo = "on_promo" in data.columns
        has_sales = "sales" in data.columns

        if not (has_price or has_promo):
            self.logger.warning("No price or promo columns found, skipping price/promo features")
            return data

        return self.price_promo_features.add_price_promo_features(
            data,
            price_column="price" if has_price else None,
            promo_column="on_promo" if has_promo else None,
            sales_column="sales" if has_sales else None,
            group_columns=group_columns,
        )

    def _build_weather_features(
        self,
        data: pd.DataFrame,
        group_columns: list[str],
    ) -> pd.DataFrame:
        """Build weather features.

        Args:
            data: DataFrame to add weather features to.
            group_columns: Columns to group by.

        Returns:
            DataFrame with weather features added.
        """
        self.logger.info("Building weather features")

        # Only include weather features if enabled in settings
        if not self.settings.include_weather_features:
            self.logger.info("Weather features disabled in settings")
            return data

        # Check if weather columns exist
        has_temp = "temperature" in data.columns
        has_precip = "precipitation" in data.columns

        if not (has_temp or has_precip):
            self.logger.warning("No weather columns found, skipping weather features")
            return data

        return self.weather_features.add_weather_features(
            data,
            temperature_column="temperature" if has_temp else None,
            precipitation_column="precipitation" if has_precip else None,
            group_columns=group_columns,
        )

    def _add_feature_flags(self, result: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Add flags indicating feature engineering operations.

        Args:
            result: DataFrame with features added.
            original_data: Original DataFrame.

        Returns:
            DataFrame with feature flags added.
        """
        # Add feature engineering metadata
        result["feature_engineered"] = True
        result["feature_engineering_date"] = pd.Timestamp.now()

        # Add feature count
        result["feature_count"] = len(result.columns)

        return result

    def _validate_features(self, result: pd.DataFrame, original_data: pd.DataFrame) -> None:
        """Validate feature engineering results.

        Args:
            result: DataFrame with features added.
            original_data: Original DataFrame.

        Raises:
            ValueError: If validation fails.
        """
        # Check that we have more features than original
        if len(result.columns) <= len(original_data.columns):
            raise ValueError("No features were added during feature engineering")

        # Check for infinite values
        numeric_columns = result.select_dtypes(include=["number"]).columns
        for col in numeric_columns:
            if result[col].isin([float("inf"), float("-inf")]).any():
                self.logger.warning(f"Found infinite values in column: {col}")

        # Check for NaN values in key columns
        key_columns = ["store_id", "item_id", "date"]
        for col in key_columns:
            if col in result.columns and result[col].isnull().any():
                raise ValueError(f"Found null values in key column: {col}")

        self.logger.info("Feature validation passed")

    def get_feature_names(
        self,
        include_calendar: bool = True,
        include_lags: bool = True,
        include_price_promo: bool = True,
        include_weather: bool = True,
    ) -> list[str]:
        """Get list of all feature names that will be generated.

        Args:
            include_calendar: Whether to include calendar feature names.
            include_lags: Whether to include lag feature names.
            include_price_promo: Whether to include price/promo feature names.
            include_weather: Whether to include weather feature names.

        Returns:
            List of feature names.
        """
        feature_names = []

        if include_calendar:
            feature_names.extend(self.calendar_features.get_calendar_feature_names())

        if include_lags:
            feature_names.extend(self.lag_features.get_lag_feature_names())
            feature_names.extend(self.lag_features.get_promo_lag_feature_names())

        if include_price_promo:
            feature_names.extend(self.price_promo_features.get_price_feature_names())
            feature_names.extend(self.price_promo_features.get_promo_feature_names())
            feature_names.extend(self.price_promo_features.get_elasticity_feature_names())
            feature_names.extend(self.price_promo_features.get_effectiveness_feature_names())

        if include_weather:
            feature_names.extend(self.weather_features.get_temperature_feature_names())
            feature_names.extend(self.weather_features.get_precipitation_feature_names())
            feature_names.extend(self.weather_features.get_weather_interaction_feature_names())

        return feature_names

    def get_feature_importance_info(self) -> dict:
        """Get information about feature importance and categories.

        Returns:
            Dictionary with feature importance information.
        """
        return {
            "calendar_features": {
                "count": len(self.calendar_features.get_calendar_feature_names()),
                "importance": "high",  # Calendar features are usually very important
                "description": "Time-based features like day of week, holidays, seasonality",
            },
            "lag_features": {
                "count": len(self.lag_features.get_lag_feature_names()) +
                        len(self.lag_features.get_promo_lag_feature_names()),
                "importance": "high",  # Lag features are crucial for time series
                "description": "Historical values and rolling statistics",
            },
            "price_promo_features": {
                "count": (len(self.price_promo_features.get_price_feature_names()) +
                         len(self.price_promo_features.get_promo_feature_names()) +
                         len(self.price_promo_features.get_elasticity_feature_names()) +
                         len(self.price_promo_features.get_effectiveness_feature_names())),
                "importance": "medium",  # Business-specific features
                "description": "Price elasticity and promotion effectiveness",
            },
            "weather_features": {
                "count": (len(self.weather_features.get_temperature_feature_names()) +
                         len(self.weather_features.get_precipitation_feature_names()) +
                         len(self.weather_features.get_weather_interaction_feature_names())),
                "importance": "low",  # External factors, may not always be available
                "description": "Weather conditions and their impact on sales",
            },
        }

    def fit_transform(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fit feature builder on training data and transform it.

        Args:
            data: Training data.
            target_column: Name of the target column.
            group_columns: Columns to group by for time series features.

        Returns:
            DataFrame with features added.
        """
        self.logger.info("Fitting and transforming features")

        # Ensure data is sorted by (store_id, item_id, date) for leakage prevention
        if group_columns is None:
            group_columns = ["store_id", "item_id"]
        
        # Validate required columns exist
        required_cols = group_columns + ["date"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for leakage prevention: {missing_cols}")

        # Sort data to ensure chronological order within groups
        sorted_data = data.sort_values(group_columns + ["date"]).reset_index(drop=True)
        
        # Build features with leakage-safe operations
        result = self._build_features_leakage_safe(
            data=sorted_data,
            target_column=target_column,
            group_columns=group_columns,
            is_training=True,
        )

        # Store feature columns for later use
        self._feature_columns = [col for col in result.columns if col not in data.columns]

        # Fit encoders and scalers if needed
        self._fit_encoders_scalers(result)

        # Mark as fitted
        self._is_fitted = True

        self.logger.info("Feature fitting completed", n_features=len(self._feature_columns))
        return result

    def transform(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        group_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Transform data using fitted feature builder.

        Args:
            data: Data to transform.
            target_column: Name of the target column.
            group_columns: Columns to group by for time series features.

        Returns:
            DataFrame with features added.
        """
        if not self._is_fitted:
            raise ValueError("Feature builder must be fitted before transform")

        self.logger.info("Transforming features")

        # Ensure data is sorted by (store_id, item_id, date) for leakage prevention
        if group_columns is None:
            group_columns = ["store_id", "item_id"]
        
        # Validate required columns exist
        required_cols = group_columns + ["date"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for leakage prevention: {missing_cols}")

        # Sort data to ensure chronological order within groups
        sorted_data = data.sort_values(group_columns + ["date"]).reset_index(drop=True)
        
        # Build features with leakage-safe operations (transform mode)
        result = self._build_features_leakage_safe(
            data=sorted_data,
            target_column=target_column,
            group_columns=group_columns,
            is_training=False,
        )

        # Apply encoders and scalers
        result = self._apply_encoders_scalers(result)

        self.logger.info("Feature transformation completed")
        return result

    def _build_features_leakage_safe(
        self,
        data: pd.DataFrame,
        target_column: str,
        group_columns: list[str],
        is_training: bool,
    ) -> pd.DataFrame:
        """Build features with strict leakage prevention.

        Args:
            data: Sorted DataFrame.
            target_column: Name of the target column.
            group_columns: Columns to group by.
            is_training: Whether this is training data.

        Returns:
            DataFrame with features added.
        """
        result = data.copy()
        
        # Calendar features (no leakage risk)
        if self.settings.include_calendar_features:
            result = self._build_calendar_features(result)

        # Lag features with strict leakage prevention
        if self.settings.include_lag_features:
            result = self._build_lag_features_leakage_safe(
                result, target_column, group_columns, is_training
            )

        # Price/promo features (no leakage risk)
        if self.settings.include_price_promo_features:
            result = self._build_price_promo_features(result, group_columns)

        # Weather features (no leakage risk)
        if self.settings.include_weather_features:
            result = self._build_weather_features(result, group_columns)

        # Add feature engineering flags
        result = self._add_feature_flags(result, data)

        # Validate features
        self._validate_features(result, data)

        return result

    def _build_lag_features_leakage_safe(
        self,
        data: pd.DataFrame,
        target_column: str,
        group_columns: list[str],
        is_training: bool,
    ) -> pd.DataFrame:
        """Build lag features with strict leakage prevention.

        Args:
            data: Sorted DataFrame.
            target_column: Name of the target column.
            group_columns: Columns to group by.
            is_training: Whether this is training data.

        Returns:
            DataFrame with lag features added.
        """
        self.logger.info("Building lag features with leakage prevention")
        
        result = data.copy()
        
        # For each group, compute lag features using only historical data
        for group_key, group_data in result.groupby(group_columns):
            group_indices = group_data.index
            
            # Compute lag features for target variable
            for lag in [1, 7, 14, 28]:
                lag_col = f"{target_column}_lag_{lag}"
                result.loc[group_indices, lag_col] = group_data[target_column].shift(lag)
            
            # Compute rolling statistics with proper window
            for window in [7, 14, 28]:
                # Rolling mean
                rolling_mean_col = f"{target_column}_rolling_mean_{window}"
                result.loc[group_indices, rolling_mean_col] = (
                    group_data[target_column].rolling(window=window, min_periods=1).mean().shift(1)
                )
                
                # Rolling std
                rolling_std_col = f"{target_column}_rolling_std_{window}"
                result.loc[group_indices, rolling_std_col] = (
                    group_data[target_column].rolling(window=window, min_periods=2).std().shift(1)
                )
            
            # EMA with shift to prevent leakage
            ema_col = f"{target_column}_ema_14"
            result.loc[group_indices, ema_col] = (
                group_data[target_column].ewm(span=14).mean().shift(1)
            )
            
            # Promo lag features if promo column exists
            if "on_promo" in group_data.columns:
                promo_lag_col = "promo_lag_1"
                result.loc[group_indices, promo_lag_col] = group_data["on_promo"].shift(1)
                
                # Promo rolling count
                promo_rolling_col = "promo_rolling_7"
                result.loc[group_indices, promo_rolling_col] = (
                    group_data["on_promo"].rolling(window=7, min_periods=1).sum().shift(1)
                )

        return result

    def _fit_encoders_scalers(self, data: pd.DataFrame) -> None:
        """Fit encoders and scalers on training data.

        Args:
            data: Training data with features.
        """
        # Identify categorical columns
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

        # Remove non-feature columns
        categorical_cols = [col for col in categorical_cols if col in self._feature_columns]

        # Fit label encoders for categorical features
        for col in categorical_cols:
            try:
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                encoder.fit(data[col].astype(str))
                self._categorical_encoders[col] = encoder
                self.logger.info(f"Fitted encoder for categorical column: {col}")
            except Exception as e:
                self.logger.warning(f"Failed to fit encoder for {col}: {e}")

    def _apply_encoders_scalers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders and scalers to data.

        Args:
            data: Data to transform.

        Returns:
            Transformed data.
        """
        result = data.copy()

        # Apply categorical encoders
        for col, encoder in self._categorical_encoders.items():
            if col in result.columns:
                try:
                    # Handle unseen categories
                    mask = result[col].isin(encoder.classes_)
                    result.loc[~mask, col] = encoder.classes_[0]  # Default to first class
                    result[col] = encoder.transform(result[col].astype(str))
                except Exception as e:
                    self.logger.warning(f"Failed to encode {col}: {e}")

        return result

    def get_feature_columns(self) -> list[str]:
        """Get list of feature columns.

        Returns:
            List of feature column names.
        """
        return self._feature_columns.copy()

    def is_fitted(self) -> bool:
        """Check if feature builder is fitted.

        Returns:
            True if fitted, False otherwise.
        """
        return self._is_fitted


def build_features(
    raw_df: pd.DataFrame,
    settings: Settings,
    target_column: str = "sales",
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convenience function to build features.

    Args:
        raw_df: Raw DataFrame.
        settings: Application settings.
        target_column: Name of the target column.
        group_columns: Columns to group by.

    Returns:
        DataFrame with features added.
    """
    builder = FeatureBuilder(settings)
    return builder.build_features(
        data=raw_df,
        target_column=target_column,
        group_columns=group_columns,
    )
