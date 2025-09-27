"""Tests for feature engineering modules."""


import pandas as pd

from forecastit.config.settings import Settings
from forecastit.features.build import FeatureBuilder
from forecastit.features.calendar import CalendarFeatures
from forecastit.features.lags import LagFeatures
from forecastit.features.price_promo import PricePromoFeatures


class TestCalendarFeatures:
    """Test calendar feature engineering."""

    def test_add_calendar_features(self):
        """Test adding calendar features."""
        calendar_features = CalendarFeatures()

        # Create test data
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "store_id": ["store_01"] * 10,
            "item_id": ["item_001"] * 10,
            "sales": [100] * 10,
        })

        # Add calendar features
        result = calendar_features.add_calendar_features(data)

        # Check that calendar features were added
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "quarter" in result.columns
        assert "is_weekend" in result.columns
        # Check for some holiday features (they may have different names)
        holiday_cols = [col for col in result.columns if "holiday" in col.lower() or "is_" in col]
        assert len(holiday_cols) > 0

        # Check data types
        assert result["day_of_week"].dtype in ["int32", "int64"]
        assert result["month"].dtype in ["int32", "int64"]
        assert result["is_weekend"].dtype == "bool"

    def test_calendar_feature_names(self):
        """Test getting calendar feature names."""
        calendar_features = CalendarFeatures()
        feature_names = calendar_features.get_calendar_feature_names()

        assert len(feature_names) > 0
        assert "day_of_week" in feature_names
        assert "month" in feature_names
        assert "is_weekend" in feature_names


class TestLagFeatures:
    """Test lag feature engineering."""

    def test_add_lag_features(self):
        """Test adding lag features."""
        lag_features = LagFeatures(max_lag=7, rolling_windows=[3, 7])

        # Create test data
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "store_id": ["store_01"] * 20,
            "item_id": ["item_001"] * 20,
            "sales": list(range(100, 120)),
        })

        # Add lag features
        result = lag_features.add_lag_features(data, target_column="sales")

        # Check that lag features were added
        assert "sales_lag_1" in result.columns
        assert "sales_lag_7" in result.columns
        assert "sales_rolling_mean_3" in result.columns
        assert "sales_rolling_mean_7" in result.columns

        # Check that first few rows have NaN for lag features
        assert pd.isna(result["sales_lag_1"].iloc[0])
        assert pd.isna(result["sales_lag_7"].iloc[0:6]).all()

    def test_lag_feature_names(self):
        """Test getting lag feature names."""
        lag_features = LagFeatures()
        feature_names = lag_features.get_lag_feature_names()

        assert len(feature_names) > 0
        assert "sales_lag_1" in feature_names
        assert "sales_rolling_mean_7" in feature_names


class TestPricePromoFeatures:
    """Test price and promotion feature engineering."""

    def test_add_price_promo_features(self):
        """Test adding price and promotion features."""
        price_promo_features = PricePromoFeatures()

        # Create test data
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "store_id": ["store_01"] * 10,
            "item_id": ["item_001"] * 10,
            "sales": [100, 120, 110, 130, 100, 150, 140, 100, 110, 120],
            "price": [10, 9, 10, 8, 10, 7, 8, 10, 10, 9],
            "on_promo": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        })

        # Add price/promo features
        result = price_promo_features.add_price_promo_features(data)

        # Check that price/promo features were added
        assert "price_pct_change" in result.columns
        assert "on_promo_rolling_7" in result.columns
        assert "price_elasticity" in result.columns

        # Check data types
        assert result["price_pct_change"].dtype == "float64"
        assert result["on_promo_rolling_7"].dtype == "float64"


class TestFeatureBuilder:
    """Test feature building orchestration."""

    def test_build_features(self):
        """Test building all features."""
        settings = Settings()
        feature_builder = FeatureBuilder(settings)

        # Create test data
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=30, freq="D"),
            "store_id": ["store_01"] * 30,
            "item_id": ["item_001"] * 30,
            "sales": list(range(100, 130)),
            "price": [10] * 30,
            "on_promo": [0] * 30,
        })

        # Build features
        result = feature_builder.build_features(data)

        # Check that features were added
        assert len(result.columns) > len(data.columns)

        # Check specific feature types
        assert "day_of_week" in result.columns  # Calendar
        assert "sales_lag_1" in result.columns  # Lag
        assert "price_pct_change" in result.columns  # Price/promo

        # Check feature flags were added
        assert "feature_engineered" in result.columns

    def test_feature_names(self):
        """Test getting feature names."""
        settings = Settings()
        feature_builder = FeatureBuilder(settings)

        feature_names = feature_builder.get_feature_names()

        assert len(feature_names) > 0
        assert "day_of_week" in feature_names
        assert "sales_lag_1" in feature_names
        assert "price_pct_change" in feature_names

    def test_feature_importance_info(self):
        """Test getting feature importance information."""
        settings = Settings()
        feature_builder = FeatureBuilder(settings)

        importance_info = feature_builder.get_feature_importance_info()

        assert "calendar_features" in importance_info
        assert "lag_features" in importance_info
        assert "price_promo_features" in importance_info
        assert "weather_features" in importance_info

        # Check structure
        for _feature_type, info in importance_info.items():
            assert "count" in info
            assert "importance" in info
            assert "description" in info
