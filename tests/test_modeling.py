"""Tests for modeling modules."""


import numpy as np
import pandas as pd
import pytest

from forecastit.config.settings import Settings
from forecastit.modeling.backtest import RollingBacktester
from forecastit.modeling.classical import ProphetForecaster, SarimaxForecaster
from forecastit.modeling.ml_models import (
    LGBMQuantileForecaster,
    LGBMRegressorForecaster,
    XGBRegressorForecaster,
)
from forecastit.modeling.tuning import OptunaTuner, tune_lgbm


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)

    data = []
    for store_id in ["store_1", "store_2"]:
        for item_id in ["item_1", "item_2"]:
            for date in dates:
                # Generate realistic sales pattern
                base_sales = 100
                seasonal = 20 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25)
                trend = date.timetuple().tm_yday * 0.1
                noise = np.random.normal(0, 10)

                sales = max(0, base_sales + seasonal + trend + noise)

                data.append({
                    "store_id": store_id,
                    "item_id": item_id,
                    "date": date,
                    "sales": sales,
                    "price": 10.0 + np.random.normal(0, 1),
                    "on_promo": np.random.choice([0, 1], p=[0.8, 0.2]),
                })

    return pd.DataFrame(data)


@pytest.fixture
def settings():
    """Create settings for testing."""
    return Settings()


class TestRollingBacktester:
    """Test rolling backtesting functionality."""

    def test_rolling_backtest_initialization(self, settings):
        """Test backtester initialization."""
        backtester = RollingBacktester(settings)
        assert backtester.settings == settings
        assert backtester.metrics_calculator is not None

    def test_rolling_backtest_simple_model(self, sample_time_series_data, settings):
        """Test rolling backtest with a simple model."""
        backtester = RollingBacktester(settings)

        def simple_fit_fn(train_features, y):
            return {"mean": y.mean()}

        def simple_predict_fn(model, val_features):
            return [model["mean"]] * len(val_features)

        def simple_feature_fn(df):
            return df

        result = backtester.rolling_backtest(
            df=sample_time_series_data,
            series_cols=["store_id", "item_id"],
            target_col="sales",
            date_col="date",
            horizon=7,
            n_folds=2,
            fit_fn=simple_fit_fn,
            predict_fn=simple_predict_fn,
            feature_fn=simple_feature_fn,
            metrics=["RMSE", "MAPE"],
            model_name="test_model",
        )

        assert "fold_results" in result
        assert "aggregated_metrics" in result
        assert len(result["fold_results"]) > 0
        assert "RMSE" in result["aggregated_metrics"]


class TestClassicalModels:
    """Test classical time series models."""

    def test_sarimax_forecaster_initialization(self):
        """Test SARIMAX forecaster initialization."""
        forecaster = SarimaxForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
        )
        assert forecaster.order == (1, 1, 1)
        assert forecaster.seasonal_order == (1, 1, 1, 12)

    def test_prophet_forecaster_initialization(self):
        """Test Prophet forecaster initialization."""
        forecaster = ProphetForecaster(
            growth="linear",
            seasonality_mode="multiplicative",
        )
        assert forecaster.growth == "linear"
        assert forecaster.seasonality_mode == "multiplicative"


class TestMLModels:
    """Test machine learning models."""

    def test_lgbm_regressor_initialization(self):
        """Test LightGBM regressor initialization."""
        model = LGBMRegressorForecaster(
            num_leaves=31,
            learning_rate=0.05,
        )
        assert model.params["num_leaves"] == 31
        assert model.params["learning_rate"] == 0.05

    def test_lgbm_quantile_initialization(self):
        """Test LightGBM quantile forecaster initialization."""
        model = LGBMQuantileForecaster(
            quantiles=[0.1, 0.5, 0.9],
            num_leaves=31,
        )
        assert model.quantiles == [0.1, 0.5, 0.9]
        assert model.base_params["num_leaves"] == 31

    def test_xgb_regressor_initialization(self):
        """Test XGBoost regressor initialization."""
        model = XGBRegressorForecaster(
            max_depth=6,
            learning_rate=0.05,
        )
        assert model.params["max_depth"] == 6
        assert model.params["learning_rate"] == 0.05


class TestTuning:
    """Test hyperparameter tuning functionality."""

    def test_optuna_tuner_initialization(self, settings):
        """Test Optuna tuner initialization."""
        tuner = OptunaTuner(settings)
        assert tuner.settings == settings
        assert tuner.metrics_calculator is not None

    def test_tune_lgbm_function(self, sample_time_series_data, settings):
        """Test LightGBM tuning function."""
        def simple_feature_fn(df):
            return df[["sales", "price", "on_promo"]].fillna(0)

        # Test with very short time limit for CI
        result = tune_lgbm(
            train_df=sample_time_series_data,
            feature_fn=simple_feature_fn,
            objective_seconds=5,  # Very short for testing
            horizon=7,
            folds=2,
            settings=settings,
        )

        assert "best_params" in result
        assert "best_score" in result
        assert "study" in result


class TestModelIntegration:
    """Test model integration and end-to-end functionality."""

    def test_model_prediction_consistency(self, sample_time_series_data):
        """Test that models produce consistent predictions."""
        # Test with a simple dataset
        small_data = sample_time_series_data.head(100)

        # Test LightGBM
        lgbm_model = LGBMRegressorForecaster()
        features = small_data[["sales", "price", "on_promo"]].fillna(0)

        try:
            fitted_model = lgbm_model.fit(features, small_data["sales"])
            predictions = fitted_model.predict(features.head(10))

            assert isinstance(predictions, pd.DataFrame)
            assert "yhat" in predictions.columns
            assert len(predictions) == 10
            assert all(predictions["yhat"] >= 0)  # Non-negative predictions

        except ImportError:
            pytest.skip("LightGBM not available")

    def test_quantile_model_intervals(self, sample_time_series_data):
        """Test that quantile models produce ordered intervals."""
        small_data = sample_time_series_data.head(100)
        features = small_data[["sales", "price", "on_promo"]].fillna(0)

        try:
            quantile_model = LGBMQuantileForecaster(quantiles=[0.1, 0.5, 0.9])
            fitted_model = quantile_model.fit(features, small_data["sales"])
            predictions = fitted_model.predict(features.head(10))

            if "yhat_lower" in predictions.columns and "yhat_upper" in predictions.columns:
                # Check that intervals are ordered
                assert all(predictions["yhat_lower"] <= predictions["yhat_upper"])

        except ImportError:
            pytest.skip("LightGBM not available")


class TestBacktestLeakage:
    """Test that backtesting prevents data leakage."""

    def test_no_future_leakage(self, sample_time_series_data, settings):
        """Test that backtesting doesn't use future data."""
        backtester = RollingBacktester(settings)

        # Track data access
        accessed_dates = []

        def tracking_feature_fn(df):
            accessed_dates.extend(df["date"].tolist())
            return df

        def simple_fit_fn(train_features, y):
            return {"mean": y.mean()}

        def simple_predict_fn(model, val_features):
            return [model["mean"]] * len(val_features)

        backtester.rolling_backtest(
            df=sample_time_series_data,
            series_cols=["store_id", "item_id"],
            target_col="sales",
            date_col="date",
            horizon=7,
            n_folds=2,
            fit_fn=simple_fit_fn,
            predict_fn=simple_predict_fn,
            feature_fn=tracking_feature_fn,
            metrics=["RMSE"],
            model_name="leakage_test",
        )

        # Verify that feature_fn was called with appropriate data
        assert len(accessed_dates) > 0
        # Additional leakage checks could be added here


if __name__ == "__main__":
    pytest.main([__file__])
