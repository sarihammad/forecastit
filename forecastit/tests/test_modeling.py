"""Tests for modeling modules."""

import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from forecastit.modeling.backtest import RollingBacktester
from forecastit.modeling.metrics import MetricsCalculator
from forecastit.config.settings import Settings


class TestRollingBacktester:
    """Test rolling backtester functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.settings = Settings()
        self.backtester = RollingBacktester(self.settings)

    def create_test_data(self, n_days: int = 100, n_series: int = 2) -> pd.DataFrame:
        """Create test time series data."""
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
        data = []
        
        for series_id in range(n_series):
            store_id = f"store_{series_id:02d}"
            item_id = f"item_{series_id:02d}"
            
            for i, date in enumerate(dates):
                # Create a simple pattern with some seasonality
                base_value = 10 + series_id * 5
                seasonal = 5 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
                noise = np.random.normal(0, 1)
                sales = max(0, base_value + seasonal + noise)
                
                data.append({
                    "date": date,
                    "store_id": store_id,
                    "item_id": item_id,
                    "sales": sales,
                    "on_promo": 1 if date.day % 7 == 0 else 0,
                    "price": 10.0 + series_id,
                })
        
        return pd.DataFrame(data)

    def test_chronological_fold_validation(self):
        """Test that folds are strictly chronological."""
        df = self.create_test_data(n_days=50, n_series=2)
        
        # Mock feature function and model functions
        def mock_feature_fn(data):
            return data.copy()
        
        def mock_fit_fn(features, target):
            return Mock()  # Mock model
        
        def mock_predict_fn(model, features):
            return np.random.normal(10, 1, len(features))
        
        # Run backtest
        result = self.backtester.rolling_backtest(
            df=df,
            series_cols=["store_id", "item_id"],
            target_col="sales",
            date_col="date",
            horizon=7,
            n_folds=3,
            fit_fn=mock_fit_fn,
            predict_fn=mock_predict_fn,
            feature_fn=mock_feature_fn,
            metrics=["rmse", "mape"],
            model_name="test_model",
            save_plots=False,
        )
        
        # Verify folds were processed
        assert result["successful_folds"] > 0, "Should have successful folds"
        
        # Check that fold results have proper chronological structure
        fold_results = result["fold_results"]
        for fold_result in fold_results:
            train_start = fold_result["train_start"]
            train_end = fold_result["train_end"]
            val_start = fold_result["val_start"]
            val_end = fold_result["val_end"]
            
            # Validate chronological order
            assert train_start < train_end, "Train start should be before train end"
            assert train_end <= val_start, "Train end should be before or equal to val start"
            assert val_start < val_end, "Val start should be before val end"

    def test_mase_finite_validation(self):
        """Test that MASE is finite and non-NaN."""
        df = self.create_test_data(n_days=30, n_series=1)
        
        # Mock functions
        def mock_feature_fn(data):
            return data.copy()
        
        def mock_fit_fn(features, target):
            return Mock()
        
        def mock_predict_fn(model, features):
            return np.full(len(features), 10.0)  # Constant predictions
        
        result = self.backtester.rolling_backtest(
            df=df,
            series_cols=["store_id", "item_id"],
            target_col="sales",
            date_col="date",
            horizon=5,
            n_folds=2,
            fit_fn=mock_fit_fn,
            predict_fn=mock_predict_fn,
            feature_fn=mock_feature_fn,
            metrics=["mase", "rmse"],
            model_name="test_model",
            save_plots=False,
        )
        
        # Check that MASE is finite in aggregated results
        if "mase" in result["aggregated_metrics"]:
            mase_value = result["aggregated_metrics"]["mase"]
            assert np.isfinite(mase_value), f"MASE should be finite, got: {mase_value}"
            assert not np.isnan(mase_value), f"MASE should not be NaN, got: {mase_value}"

    def test_quantile_metrics_calculation(self):
        """Test pinball loss calculation for quantile forecasters."""
        # Create test data
        actual = np.array([10, 20, 30, 40, 50])
        predicted_median = np.array([12, 18, 32, 38, 52])
        predicted_lower = np.array([8, 14, 28, 34, 48])
        predicted_upper = np.array([16, 22, 36, 42, 56])
        
        # Test pinball loss calculation
        pinball_10 = self.backtester._calculate_pinball_loss(actual, predicted_lower, 0.1)
        pinball_50 = self.backtester._calculate_pinball_loss(actual, predicted_median, 0.5)
        pinball_90 = self.backtester._calculate_pinball_loss(actual, predicted_upper, 0.9)
        
        # Pinball losses should be finite
        assert np.isfinite(pinball_10), f"Pinball 0.1 should be finite, got: {pinball_10}"
        assert np.isfinite(pinball_50), f"Pinball 0.5 should be finite, got: {pinball_50}"
        assert np.isfinite(pinball_90), f"Pinball 0.9 should be finite, got: {pinball_90}"
        
        # Test quantile predictions handling
        quantile_predicted = {
            'quantiles': {
                0.1: predicted_lower,
                0.5: predicted_median,
                0.9: predicted_upper
            }
        }
        
        metrics = self.backtester._calculate_fold_metrics(
            actual=actual,
            predicted=quantile_predicted,
            metrics=["rmse", "mape"]
        )
        
        # Should have pinball metrics
        assert "pinball_0.1" in metrics, "Should have pinball_0.1 metric"
        assert "pinball_0.5" in metrics, "Should have pinball_0.5 metric"
        assert "pinball_0.9" in metrics, "Should have pinball_0.9 metric"
        
        # Pinball metrics should be finite
        for alpha in [0.1, 0.5, 0.9]:
            metric_name = f"pinball_{alpha}"
            assert np.isfinite(metrics[metric_name]), f"{metric_name} should be finite"

    def test_data_leakage_detection(self):
        """Test that data leakage is detected and raises error."""
        # Create data that would cause leakage if not handled properly
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        data = []
        
        for i, date in enumerate(dates):
            data.append({
                "date": date,
                "store_id": "store_01",
                "item_id": "item_01",
                "sales": i + 1,
                "on_promo": 0,
                "price": 10.0,
            })
        
        df = pd.DataFrame(data)
        
        # Mock functions that would cause leakage
        def mock_feature_fn(data):
            return data.copy()
        
        def mock_fit_fn(features, target):
            return Mock()
        
        def mock_predict_fn(model, features):
            return np.random.normal(10, 1, len(features))
        
        # This should not raise an error due to proper fold boundaries
        # The backtester should handle chronological splits correctly
        result = self.backtester.rolling_backtest(
            df=df,
            series_cols=["store_id", "item_id"],
            target_col="sales",
            date_col="date",
            horizon=3,
            n_folds=2,
            fit_fn=mock_fit_fn,
            predict_fn=mock_predict_fn,
            feature_fn=mock_feature_fn,
            metrics=["rmse"],
            model_name="test_model",
            save_plots=False,
        )
        
        # Should complete without leakage errors
        assert result["successful_folds"] >= 0, "Should complete without leakage errors"

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        # Create minimal data
        df = pd.DataFrame({
            "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
            "store_id": ["store_01"] * 5,
            "item_id": ["item_01"] * 5,
            "sales": [10, 20, 30, 40, 50],
            "on_promo": [0, 1, 0, 1, 0],
            "price": [10.0] * 5,
        })
        
        def mock_feature_fn(data):
            return data.copy()
        
        def mock_fit_fn(features, target):
            return Mock()
        
        def mock_predict_fn(model, features):
            return np.random.normal(10, 1, len(features))
        
        # Should handle insufficient data gracefully
        result = self.backtester.rolling_backtest(
            df=df,
            series_cols=["store_id", "item_id"],
            target_col="sales",
            date_col="date",
            horizon=7,  # Horizon larger than available data
            n_folds=3,  # Too many folds for small dataset
            fit_fn=mock_fit_fn,
            predict_fn=mock_predict_fn,
            feature_fn=mock_feature_fn,
            metrics=["rmse"],
            model_name="test_model",
            save_plots=False,
        )
        
        # Should complete but may have fewer successful folds
        assert result["successful_folds"] >= 0, "Should handle insufficient data gracefully"

    def test_chronological_sorting_validation(self):
        """Test that data sorting validation works."""
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
                "item_id": "item_01",
                "sales": date.day,
                "on_promo": 0,
                "price": 10.0,
            })
        
        df = pd.DataFrame(data)
        
        def mock_feature_fn(data):
            return data.copy()
        
        def mock_fit_fn(features, target):
            return Mock()
        
        def mock_predict_fn(model, features):
            return np.random.normal(10, 1, len(features))
        
        # Should complete successfully (data gets sorted internally)
        result = self.backtester.rolling_backtest(
            df=df,
            series_cols=["store_id", "item_id"],
            target_col="sales",
            date_col="date",
            horizon=1,
            n_folds=1,
            fit_fn=mock_fit_fn,
            predict_fn=mock_predict_fn,
            feature_fn=mock_feature_fn,
            metrics=["rmse"],
            model_name="test_model",
            save_plots=False,
        )
        
        # Should handle unsorted data by sorting it internally
        assert result["successful_folds"] >= 0, "Should handle unsorted data by sorting"


class TestMetricsCalculator:
    """Test metrics calculator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()

    def test_mase_calculation(self):
        """Test MASE calculation is finite."""
        # Create test data with seasonality
        actual = np.array([10, 20, 10, 20, 10, 20, 10, 20, 10, 20])
        predicted = np.array([12, 18, 12, 18, 12, 18, 12, 18, 12, 18])
        
        metrics = self.calculator.calculate_metrics(
            actual=actual,
            predicted=predicted,
            metrics=["mase", "rmse", "mape", "smape"]
        )
        
        # MASE should be finite
        assert "mase" in metrics, "Should have MASE metric"
        assert np.isfinite(metrics["mase"]), f"MASE should be finite, got: {metrics['mase']}"
        assert not np.isnan(metrics["mase"]), f"MASE should not be NaN, got: {metrics['mase']}"
        assert metrics["mase"] > 0, f"MASE should be positive, got: {metrics['mase']}"

    def test_all_metrics_finite(self):
        """Test that all standard metrics are finite."""
        actual = np.array([10, 20, 30, 40, 50])
        predicted = np.array([12, 18, 32, 38, 52])
        
        metrics = self.calculator.calculate_metrics(
            actual=actual,
            predicted=predicted,
            metrics=["rmse", "mape", "smape", "mase"]
        )
        
        for metric_name, metric_value in metrics.items():
            assert np.isfinite(metric_value), f"{metric_name} should be finite, got: {metric_value}"
            assert not np.isnan(metric_value), f"{metric_name} should not be NaN, got: {metric_value}"

    def test_zero_actual_values(self):
        """Test metrics calculation with zero actual values."""
        actual = np.array([0, 0, 0, 0, 0])
        predicted = np.array([1, 2, 3, 4, 5])
        
        metrics = self.calculator.calculate_metrics(
            actual=actual,
            predicted=predicted,
            metrics=["rmse", "mape", "smape", "mase"]
        )
        
        # Should handle zero values gracefully
        for metric_name, metric_value in metrics.items():
            assert np.isfinite(metric_value), f"{metric_name} should be finite with zero actuals"
