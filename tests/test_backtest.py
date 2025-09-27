"""Tests for backtesting functionality."""

import pandas as pd
import numpy as np
from forecastit.modeling.backtest import RollingBacktester
from forecastit.config.settings import Settings


class DummyModel:
    """Simple dummy model for testing."""
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return pd.DataFrame({
            "date": X["date"], 
            "yhat": np.zeros(len(X))
        })


class DummyFeatureBuilder:
    """Simple dummy feature builder for testing."""
    
    def fit_transform(self, df):
        return df
    
    def transform(self, df):
        return df


def test_backtest_strict_and_metrics(tmp_path):
    """Test backtesting with strict chronological validation and metrics."""
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "date": dates, 
        "store_id": "S", 
        "item_id": "I", 
        "sales": np.arange(40)
    })

    def fit_fn(train):
        return DummyModel().fit(train)
    
    def predict_fn(model, X):
        return model.predict(X)

    settings = Settings()
    bt = RollingBacktester(settings)
    
    res = bt.rolling_backtest(
        df=df, 
        series_cols=["store_id", "item_id"], 
        target_col="sales", 
        date_col="date",
        horizon=7, 
        n_folds=2, 
        fit_fn=fit_fn, 
        predict_fn=predict_fn, 
        feature_fn=DummyFeatureBuilder(),
        metrics=["rmse", "mape", "smape", "mase"]
    )
    
    assert "aggregated" in res
    assert all(k in res["aggregated"] for k in ["rmse", "mape", "smape", "mase"])


def test_backtest_quantile_metrics(tmp_path):
    """Test backtesting with quantile models and pinball loss."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates, 
        "store_id": "S", 
        "item_id": "I", 
        "sales": np.random.randn(30) + 10
    })

    class DummyQuantileModel:
        def fit(self, X, y=None):
            return self
        
        def predict(self, X):
            return pd.DataFrame({
                "date": X["date"],
                "yhat_0.1": np.ones(len(X)) * 8,
                "yhat_0.5": np.ones(len(X)) * 10,
                "yhat_0.9": np.ones(len(X)) * 12,
            })

    def fit_fn(train):
        return DummyQuantileModel().fit(train)
    
    def predict_fn(model, X):
        return model.predict(X)

    settings = Settings()
    bt = RollingBacktester(settings)
    
    res = bt.rolling_backtest(
        df=df, 
        series_cols=["store_id", "item_id"], 
        target_col="sales", 
        date_col="date",
        horizon=5, 
        n_folds=2, 
        fit_fn=fit_fn, 
        predict_fn=predict_fn, 
        feature_fn=DummyFeatureBuilder(),
        metrics=["rmse", "pinball_0.1", "pinball_0.5", "pinball_0.9"]
    )
    
    assert "aggregated" in res
    assert "pinball_0.1" in res["aggregated"]
    assert "pinball_0.5" in res["aggregated"]
    assert "pinball_0.9" in res["aggregated"]
