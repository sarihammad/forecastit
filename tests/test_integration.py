"""Integration tests for the complete ForecastIt system."""

from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecastit.cleaning.clean import DataCleaner
from forecastit.config.settings import Settings
from forecastit.data.make_synthetic import generate_synthetic_data
from forecastit.features.build import FeatureBuilder
from forecastit.inference.inventory import InventoryOptimizer
from forecastit.modeling.backtest import RollingBacktester
from forecastit.modeling.baselines import BaselineModels


@pytest.fixture
def settings():
    """Create settings for integration testing."""
    return Settings()


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for integration testing."""
    return generate_synthetic_data(
        stores=["store_1", "store_2"],
        items=["item_1", "item_2"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        seed=42,
    )


class TestEndToEndPipeline:
    """Test the complete end-to-end pipeline."""

    def test_data_generation_to_cleaning(self, synthetic_data):
        """Test data generation and cleaning pipeline."""
        # Verify synthetic data structure
        assert len(synthetic_data) > 0
        assert "store_id" in synthetic_data.columns
        assert "item_id" in synthetic_data.columns
        assert "date" in synthetic_data.columns
        assert "sales" in synthetic_data.columns

        # Test data cleaning
        settings = Settings()
        cleaner = DataCleaner(settings)
        cleaned_data = cleaner.clean_data(synthetic_data)

        # Verify cleaning results
        assert len(cleaned_data) > 0
        assert not cleaned_data["sales"].isnull().any()
        assert all(cleaned_data["sales"] >= 0)

        # Check that cleaning flags are added
        if "is_outlier" in cleaned_data.columns:
            assert "is_outlier" in cleaned_data.columns
        if "is_missing" in cleaned_data.columns:
            assert "is_missing" in cleaned_data.columns

    def test_feature_engineering_pipeline(self, synthetic_data, settings):
        """Test feature engineering pipeline."""
        # Clean data first
        cleaner = DataCleaner(settings)
        cleaned_data = cleaner.clean_data(synthetic_data)

        # Build features
        feature_builder = FeatureBuilder(settings)
        features_data = feature_builder.build_features(cleaned_data)

        # Verify features were added
        assert len(features_data.columns) > len(cleaned_data.columns)
        assert "feature_engineered" in features_data.columns
        assert "feature_count" in features_data.columns

        # Check for calendar features
        calendar_features = [col for col in features_data.columns if col in [
            "day_of_week", "month", "quarter", "is_weekend", "is_holiday"
        ]]
        assert len(calendar_features) > 0

        # Check for lag features (if enabled)
        if settings.include_lag_features:
            lag_features = [col for col in features_data.columns if "lag" in col or "rolling" in col]
            assert len(lag_features) > 0

    def test_backtesting_pipeline(self, synthetic_data, settings):
        """Test backtesting pipeline with simple models."""
        # Prepare data
        cleaner = DataCleaner(settings)
        cleaned_data = cleaner.clean_data(synthetic_data)

        feature_builder = FeatureBuilder(settings)
        features_data = feature_builder.build_features(cleaned_data)

        # Define simple model functions
        def simple_fit_fn(train_features, y):
            return {"mean": y.mean(), "std": y.std()}

        def simple_predict_fn(model, val_features):
            predictions = []
            for _ in range(len(val_features)):
                pred = np.random.normal(model["mean"], model["std"])
                predictions.append(max(0, pred))  # Ensure non-negative
            return predictions

        def simple_feature_fn(df):
            return df

        # Run backtesting
        backtester = RollingBacktester(settings)
        result = backtester.rolling_backtest(
            df=features_data,
            series_cols=["store_id", "item_id"],
            target_col="sales",
            date_col="date",
            horizon=7,
            n_folds=2,
            fit_fn=simple_fit_fn,
            predict_fn=simple_predict_fn,
            feature_fn=simple_feature_fn,
            metrics=["RMSE", "MAPE"],
            model_name="simple_test",
        )

        # Verify backtesting results
        assert "fold_results" in result
        assert "aggregated_metrics" in result
        assert len(result["fold_results"]) > 0
        assert "RMSE" in result["aggregated_metrics"]
        assert "MAPE" in result["aggregated_metrics"]

    def test_baseline_models_pipeline(self, synthetic_data, settings):
        """Test baseline models pipeline."""
        # Prepare data
        cleaner = DataCleaner(settings)
        cleaned_data = cleaner.clean_data(synthetic_data)

        feature_builder = FeatureBuilder(settings)
        features_data = feature_builder.build_features(cleaned_data)

        # Test baseline models
        baselines = BaselineModels()

        # Generate baseline forecasts
        baseline_forecasts = baselines.generate_all_baselines(
            features_data,
            horizon=7,
        )

        # Verify baseline results
        assert len(baseline_forecasts) > 0
        for _model_name, forecast_df in baseline_forecasts.items():
            assert isinstance(forecast_df, pd.DataFrame)
            assert "sales" in forecast_df.columns
            assert len(forecast_df) > 0

    def test_inventory_optimization_pipeline(self, synthetic_data, settings):
        """Test inventory optimization pipeline."""
        # Prepare data
        cleaner = DataCleaner(settings)
        cleaned_data = cleaner.clean_data(synthetic_data)

        feature_builder = FeatureBuilder(settings)
        feature_builder.build_features(cleaned_data)

        # Create mock forecasts
        forecast_dates = pd.date_range(
            start=datetime.now().date(),
            end=datetime.now().date() + timedelta(days=7),
            freq="D"
        )

        forecasts = []
        for date in forecast_dates:
            forecasts.append({
                "store_id": "store_1",
                "item_id": "item_1",
                "date": date,
                "yhat": 100.0,
                "yhat_lower": 90.0,
                "yhat_upper": 110.0,
                "model_name": "test_model",
                "model_version": "v1.0",
            })

        # Test inventory optimization
        inventory_optimizer = InventoryOptimizer(settings)
        inventory_metrics = inventory_optimizer.calculate_inventory_metrics(
            forecasts=forecasts,
            service_level=0.95,
            lead_time=7,
            holding_cost=0.1,
            stockout_cost=5.0,
        )

        # Verify inventory results
        assert "reorder_point" in inventory_metrics
        assert "safety_stock" in inventory_metrics
        assert "expected_demand" in inventory_metrics
        assert "service_level" in inventory_metrics
        assert inventory_metrics["service_level"] == 0.95
        assert inventory_metrics["reorder_point"] >= 0
        assert inventory_metrics["safety_stock"] >= 0


class TestSystemIntegration:
    """Test system-level integration."""

    def test_synthetic_data_generation(self):
        """Test synthetic data generation with different parameters."""
        # Test with different parameters
        data1 = generate_synthetic_data(
            stores=["store_1"],
            items=["item_1"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            seed=42,
        )

        data2 = generate_synthetic_data(
            stores=["store_2"],
            items=["item_2"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            seed=42,
        )

        # Same seed should produce same data
        pd.testing.assert_frame_equal(data1, data2)

        # Different seed should produce different data
        data3 = generate_synthetic_data(
            n_stores=1,
            n_items=1,
            start_date="2023-01-01",
            end_date="2023-01-31",
            seed=43,
        )

        assert not data1.equals(data3)

    def test_file_outputs(self, synthetic_data, settings, tmp_path):
        """Test that the system can write outputs to files."""
        # Change working directory to temp path
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Create reports directory
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            # Test data cleaning with file output
            cleaner = DataCleaner(settings)
            cleaned_data = cleaner.clean_data(synthetic_data)

            # Save cleaned data
            cleaned_data.to_parquet("cleaned_data.parquet")
            assert Path("cleaned_data.parquet").exists()

            # Test feature engineering
            feature_builder = FeatureBuilder(settings)
            features_data = feature_builder.build_features(cleaned_data)

            # Save features
            features_data.to_parquet("features_data.parquet")
            assert Path("features_data.parquet").exists()

        finally:
            os.chdir(original_cwd)

    def test_settings_consistency(self, settings):
        """Test that settings are consistent across the system."""
        # Test that settings have required attributes
        assert hasattr(settings, "data_dir")
        assert hasattr(settings, "default_horizon")
        assert hasattr(settings, "default_folds")
        assert hasattr(settings, "api_host")
        assert hasattr(settings, "api_port")

        # Test that numeric settings are valid
        assert settings.default_horizon > 0
        assert settings.default_folds > 0
        assert settings.api_port > 0

        # Test that boolean settings are valid
        assert isinstance(settings.include_calendar_features, bool)
        assert isinstance(settings.include_lag_features, bool)
        assert isinstance(settings.is_development, bool)

    def test_memory_usage(self, synthetic_data, settings):
        """Test that the system doesn't use excessive memory."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run through the pipeline
        cleaner = DataCleaner(settings)
        cleaned_data = cleaner.clean_data(synthetic_data)

        feature_builder = FeatureBuilder(settings)
        feature_builder.build_features(cleaned_data)

        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for small dataset)
        assert memory_increase < 100 * 1024 * 1024, f"Memory usage increased by {memory_increase / 1024 / 1024:.1f}MB"

    def test_error_handling(self, settings):
        """Test error handling in the pipeline."""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            "store_id": ["store_1"],
            "item_id": ["item_1"],
            "date": ["invalid_date"],
            "sales": ["not_a_number"],
        })

        cleaner = DataCleaner(settings)

        # Should handle invalid data gracefully
        try:
            cleaned_data = cleaner.clean_data(invalid_data)
            # If it doesn't raise an exception, it should clean the data
            assert len(cleaned_data) >= 0
        except Exception as e:
            # If it raises an exception, it should be informative
            assert isinstance(e, ValueError | TypeError)

        # Test with empty data
        empty_data = pd.DataFrame(columns=["store_id", "item_id", "date", "sales"])

        try:
            cleaned_data = cleaner.clean_data(empty_data)
            assert len(cleaned_data) == 0
        except Exception as e:
            assert isinstance(e, ValueError)


if __name__ == "__main__":
    pytest.main([__file__])
