"""Tests for feature engineering."""

import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta

from forecastit.features.build import FeatureBuilder
from forecastit.config.settings import Settings


class TestFeatureBuilder:
    """Test feature builder functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = Settings()
        self.settings.include_calendar_features = True
        self.settings.include_lag_features = True
        self.settings.include_price_promo_features = True
        self.settings.include_weather_features = False
        
        self.builder = FeatureBuilder(self.settings)

    def test_leakage_prevention_fit_transform(self):
        """Test that fit_transform prevents data leakage."""
        # Create test data with known patterns
        dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
        
        data = []
        for store_id in ["store_01", "store_02"]:
            for item_id in ["item_001", "item_002"]:
                for i, date in enumerate(dates):
                    # Create a simple pattern: sales = day_of_month + store_hash + item_hash
                    base_sales = (date.day + hash(store_id) % 10 + hash(item_id) % 10) % 50
                    data.append({
                        "date": date,
                        "store_id": store_id,
                        "item_id": item_id,
                        "sales": base_sales,
                        "on_promo": 1 if date.day % 7 == 0 else 0,
                        "price": 10.0 + (hash(item_id) % 5),
                    })
        
        df = pd.DataFrame(data)
        
        # Fit and transform
        result = self.builder.fit_transform(df, target_column="sales")
        
        # Test 1: Check that lag features never equal same-day target
        for lag in [1, 7, 14, 28]:
            lag_col = f"sales_lag_{lag}"
            if lag_col in result.columns:
                # Find rows where lag could theoretically equal target
                mask = result["sales_lag_1"].notna() & result["sales"].notna()
                if mask.any():
                    # For lag=1, this should never happen due to leakage prevention
                    same_values = (result.loc[mask, "sales_lag_1"] == result.loc[mask, "sales"]).sum()
                    assert same_values == 0, f"Found {same_values} cases where lag_1 equals target (leakage!)"
        
        # Test 2: Check that rolling features are shifted
        for window in [7, 14, 28]:
            rolling_mean_col = f"sales_rolling_mean_{window}"
            if rolling_mean_col in result.columns:
                # Rolling mean should not equal current value (due to shift(1))
                mask = result[rolling_mean_col].notna() & result["sales"].notna()
                if mask.any():
                    # Should be very few exact matches due to shift
                    exact_matches = (result.loc[mask, rolling_mean_col] == result.loc[mask, "sales"]).sum()
                    assert exact_matches <= 2, f"Too many exact matches in rolling mean (potential leakage): {exact_matches}"

    def test_leakage_prevention_transform(self):
        """Test that transform prevents data leakage on validation data."""
        # Create training data
        train_dates = pd.date_range(start="2023-01-01", end="2023-01-15", freq="D")
        train_data = []
        for store_id in ["store_01"]:
            for item_id in ["item_001"]:
                for date in train_dates:
                    train_data.append({
                        "date": date,
                        "store_id": store_id,
                        "item_id": item_id,
                        "sales": date.day * 2,
                        "on_promo": 1 if date.day % 7 == 0 else 0,
                        "price": 10.0,
                    })
        
        train_df = pd.DataFrame(train_data)
        
        # Fit on training data
        self.builder.fit_transform(train_df, target_column="sales")
        
        # Create validation data (future dates)
        val_dates = pd.date_range(start="2023-01-16", end="2023-01-20", freq="D")
        val_data = []
        for store_id in ["store_01"]:
            for item_id in ["item_001"]:
                for date in val_dates:
                    val_data.append({
                        "date": date,
                        "store_id": store_id,
                        "item_id": item_id,
                        "sales": date.day * 2,  # Same pattern
                        "on_promo": 1 if date.day % 7 == 0 else 0,
                        "price": 10.0,
                    })
        
        val_df = pd.DataFrame(val_data)
        
        # Transform validation data
        val_result = self.builder.transform(val_df, target_column="sales")
        
        # Test: Check that lag features in validation set are NaN for first few rows
        # (since we don't have historical data for these future dates)
        lag_col = "sales_lag_1"
        if lag_col in val_result.columns:
            # First row should have NaN for lag features (no historical data)
            first_row_lag = val_result.iloc[0][lag_col]
            assert pd.isna(first_row_lag), "First validation row should have NaN lag features"
        
        # Test: Rolling features should also be NaN for early rows
        rolling_mean_col = "sales_rolling_mean_7"
        if rolling_mean_col in val_result.columns:
            # First few rows should have NaN for rolling features
            first_few_nans = val_result[rolling_mean_col].head(3).isna().sum()
            assert first_few_nans >= 1, "First few validation rows should have NaN rolling features"

    def test_chronological_sorting(self):
        """Test that data is properly sorted chronologically."""
        # Create unsorted data
        dates = [
            datetime(2023, 1, 3),
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
        ]
        
        data = []
        for date in dates:
            data.append({
                "date": date,
                "store_id": "store_01",
                "item_id": "item_001",
                "sales": date.day,
                "on_promo": 0,
                "price": 10.0,
            })
        
        df = pd.DataFrame(data)
        
        # Fit and transform
        result = self.builder.fit_transform(df, target_column="sales")
        
        # Check that dates are sorted
        sorted_dates = result["date"].tolist()
        assert sorted_dates == sorted(sorted_dates), "Dates should be sorted chronologically"

    def test_feature_columns_tracking(self):
        """Test that feature columns are properly tracked."""
        # Create simple test data
        dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
        data = []
        for date in dates:
            data.append({
                "date": date,
                "store_id": "store_01",
                "item_id": "item_001",
                "sales": 10,
                "on_promo": 0,
                "price": 10.0,
            })
        
        df = pd.DataFrame(data)
        original_columns = set(df.columns)
        
        # Fit and transform
        result = self.builder.fit_transform(df, target_column="sales")
        
        # Check that feature columns are tracked
        feature_columns = set(self.builder.get_feature_columns())
        result_columns = set(result.columns)
        new_columns = result_columns - original_columns
        
        assert feature_columns == new_columns, "Feature columns should match new columns"
        assert len(feature_columns) > 0, "Should have created some features"

    def test_fitted_state(self):
        """Test fitted state tracking."""
        # Initially not fitted
        assert not self.builder.is_fitted(), "Should not be fitted initially"
        
        # Create test data
        dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
        data = []
        for date in dates:
            data.append({
                "date": date,
                "store_id": "store_01",
                "item_id": "item_001",
                "sales": 10,
                "on_promo": 0,
                "price": 10.0,
            })
        
        df = pd.DataFrame(data)
        
        # Fit and transform
        self.builder.fit_transform(df, target_column="sales")
        
        # Should be fitted now
        assert self.builder.is_fitted(), "Should be fitted after fit_transform"
        
        # Transform should work
        val_result = self.builder.transform(df, target_column="sales")
        assert len(val_result) == len(df), "Transform should work on fitted builder"

    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises appropriate error."""
        dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
        data = []
        for date in dates:
            data.append({
                "date": date,
                "store_id": "store_01",
                "item_id": "item_001",
                "sales": 10,
                "on_promo": 0,
                "price": 10.0,
            })
        
        df = pd.DataFrame(data)
        
        # Should raise error when trying to transform without fitting
        with pytest.raises(ValueError, match="Feature builder must be fitted before transform"):
            self.builder.transform(df, target_column="sales")

    def test_missing_required_columns_raises_error(self):
        """Test that missing required columns raise appropriate error."""
        # Create data without required columns
        df = pd.DataFrame({
            "sales": [10, 20, 30],
            "on_promo": [0, 1, 0],
        })
        
        # Should raise error when required columns are missing
        with pytest.raises(ValueError, match="Missing required columns for leakage prevention"):
            self.builder.fit_transform(df, target_column="sales")
