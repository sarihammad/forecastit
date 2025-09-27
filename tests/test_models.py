"""Tests for modeling modules."""


import numpy as np
import pandas as pd

from forecastit.modeling.baselines import BaselineModels
from forecastit.modeling.datasets import TimeSeriesDataset
from forecastit.modeling.metrics import MetricsCalculator


class TestTimeSeriesDataset:
    """Test time series dataset handling."""

    def test_dataset_initialization(self):
        """Test dataset initialization."""
        # Create test data
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "store_id": ["store_01"] * 10,
            "item_id": ["item_001"] * 10,
            "sales": list(range(100, 110)),
        })

        dataset = TimeSeriesDataset(data)

        assert dataset.target_column == "sales"
        assert dataset.group_columns == ["store_id", "item_id"]
        assert len(dataset.data) == 10

    def test_split_by_date(self):
        """Test splitting dataset by date."""
        # Create test data
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "store_id": ["store_01"] * 20,
            "item_id": ["item_001"] * 20,
            "sales": list(range(100, 120)),
        })

        dataset = TimeSeriesDataset(data)

        # Split data
        train, val, test = dataset.split_by_date(
            train_end_date="2023-01-10",
            val_start_date="2023-01-11",
            test_start_date="2023-01-16",
        )

        assert len(train) == 10
        assert len(val) == 5
        assert len(test) == 5

        # Check date ranges
        assert train["date"].max() <= pd.to_datetime("2023-01-10")
        assert val["date"].min() >= pd.to_datetime("2023-01-11")
        assert test["date"].min() >= pd.to_datetime("2023-01-16")

    def test_prepare_for_modeling(self):
        """Test preparing data for modeling."""
        # Create test data with features
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "store_id": ["store_01"] * 10,
            "item_id": ["item_001"] * 10,
            "sales": list(range(100, 110)),
            "feature1": list(range(10)),
            "feature2": list(range(10, 20)),
        })

        dataset = TimeSeriesDataset(data)

        X, y = dataset.prepare_for_modeling()

        assert X.shape[0] == 10
        assert X.shape[1] == 2  # feature1 and feature2
        assert y.shape[0] == 10
        assert np.array_equal(y, data["sales"].values)

    def test_validate_data(self):
        """Test data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "store_id": ["store_01"] * 10,
            "item_id": ["item_001"] * 10,
            "sales": list(range(100, 110)),
        })

        dataset = TimeSeriesDataset(valid_data)
        assert dataset.validate_data() is True

        # Invalid data (missing column)
        invalid_data = valid_data.drop(columns=["sales"])
        dataset_invalid = TimeSeriesDataset(invalid_data)
        assert dataset_invalid.validate_data() is False


class TestMetricsCalculator:
    """Test metrics calculation."""

    def test_calculate_metrics(self):
        """Test calculating multiple metrics."""
        calculator = MetricsCalculator()

        # Create test data
        y_true = np.array([100, 110, 120, 130, 140])
        y_pred = np.array([105, 115, 125, 135, 145])

        metrics = calculator.calculate_metrics(y_true, y_pred)

        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "MAPE" in metrics
        assert "sMAPE" in metrics
        assert "R2" in metrics

        # Check that metrics are reasonable
        assert metrics["RMSE"] > 0
        assert metrics["MAE"] > 0
        assert metrics["R2"] > 0  # Should be positive for this simple case

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        calculator = MetricsCalculator()

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        rmse = calculator._rmse(y_true, y_pred)

        # RMSE should be approximately 0.1 for this case
        assert abs(rmse - 0.1) < 0.01

    def test_mape_calculation(self):
        """Test MAPE calculation."""
        calculator = MetricsCalculator()

        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 180, 330])

        mape = calculator._mape(y_true, y_pred)

        # MAPE should be approximately 10% for this case
        assert abs(mape - 10.0) < 1.0

    def test_series_metrics(self):
        """Test calculating metrics per series."""
        calculator = MetricsCalculator()

        # Create test data with multiple series
        data = pd.DataFrame({
            "store_id": ["store_01", "store_01", "store_02", "store_02"],
            "item_id": ["item_001", "item_001", "item_001", "item_001"],
            "y_true": [100, 110, 120, 130],
            "y_pred": [105, 115, 125, 135],
        })

        series_metrics = calculator.calculate_series_metrics(
            data=data,
            y_true_column="y_true",
            y_pred_column="y_pred",
            group_columns=["store_id", "item_id"],
        )

        assert len(series_metrics) == 2  # Two series
        assert "RMSE" in series_metrics.columns
        assert "MAE" in series_metrics.columns


class TestBaselineModels:
    """Test baseline forecasting models."""

    def test_naive_forecast(self):
        """Test naive forecasting."""
        baselines = BaselineModels()

        # Create test data
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "store_id": ["store_01"] * 10,
            "item_id": ["item_001"] * 10,
            "sales": list(range(100, 110)),
        })

        forecasts = baselines.naive_forecast(data, horizon=5)

        assert len(forecasts) == 5
        assert forecasts["sales"].iloc[0] == 109  # Last value
        assert all(forecasts["forecast_type"] == "naive")

    def test_seasonal_naive_forecast(self):
        """Test seasonal naive forecasting."""
        baselines = BaselineModels()

        # Create test data with weekly pattern
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=14, freq="D"),
            "store_id": ["store_01"] * 14,
            "item_id": ["item_001"] * 14,
            "sales": [100, 110, 120, 130, 140, 150, 160] * 2,  # Weekly pattern
        })

        forecasts = baselines.seasonal_naive_forecast(
            data, horizon=7, seasonal_period=7
        )

        assert len(forecasts) == 7
        assert all(forecasts["forecast_type"] == "seasonal_naive")

    def test_moving_average_forecast(self):
        """Test moving average forecasting."""
        baselines = BaselineModels()

        # Create test data
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "store_id": ["store_01"] * 10,
            "item_id": ["item_001"] * 10,
            "sales": list(range(100, 110)),
        })

        forecasts = baselines.moving_average_forecast(data, horizon=5, window=3)

        assert len(forecasts) == 5
        assert all(forecasts["forecast_type"] == "moving_average")
        # Should be average of last 3 values
        expected_value = (107 + 108 + 109) / 3
        assert abs(forecasts["sales"].iloc[0] - expected_value) < 0.01

    def test_generate_all_baselines(self):
        """Test generating all baseline forecasts."""
        baselines = BaselineModels()

        # Create test data
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "store_id": ["store_01"] * 10,
            "item_id": ["item_001"] * 10,
            "sales": list(range(100, 110)),
        })

        all_forecasts = baselines.generate_all_baselines(data, horizon=5)

        assert "naive" in all_forecasts
        assert "seasonal_naive" in all_forecasts
        assert "moving_average" in all_forecasts
        assert "exponential_smoothing" in all_forecasts

        # Each should have 5 forecasts
        for model_name, forecasts in all_forecasts.items():
            assert len(forecasts) == 5
            assert all(forecasts["forecast_type"] == model_name)
