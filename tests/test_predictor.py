"""Tests for the Predictor class to hit quantile and non-quantile branches."""

import types
import numpy as np
import pandas as pd
from unittest.mock import patch
from forecastit.inference.predictor import Predictor


class DummyQuantileModel:
    """Dummy model that returns quantile predictions."""
    
    def predict(self, X):
        # return df-like with yhat,yhat_lower,yhat_upper
        return pd.DataFrame({
            "date": X["date"].values if "date" in X.columns else pd.date_range("2023-01-01", periods=len(X)),
            "yhat": np.ones(len(X)) * 10,
            "yhat_lower": np.ones(len(X)) * 8,
            "yhat_upper": np.ones(len(X)) * 12,
        })


class DummyPointModel:
    """Dummy model that returns only point predictions."""
    
    def predict(self, X):
        return pd.DataFrame({
            "date": X["date"].values if "date" in X.columns else pd.date_range("2023-01-01", periods=len(X)),
            "yhat": np.ones(len(X)) * 7
        })


def test_predictor_quantile(monkeypatch):
    """Test predictor with quantile model that provides intervals."""
    dummy = types.SimpleNamespace(
        preprocessor=types.SimpleNamespace(transform=lambda df: df),
        model=DummyQuantileModel(),
        meta={"model_name": "lgbm_quantile", "version": "1"},
    )
    
    # monkeypatch registry loader to return dummy
    from forecastit.modeling.registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "load_champion", lambda *a, **k: (dummy.preprocessor, dummy.model, dummy.meta))

    pr = Predictor()
    df = pr.predict(store_id="S1", item_id="I1", start_date="2023-01-01", end_date="2023-01-03")
    assert {"yhat", "yhat_lower", "yhat_upper"}.issubset(df.columns)


def test_predictor_fallback_intervals(monkeypatch):
    """Test predictor with point model that needs fallback intervals."""
    dummy = types.SimpleNamespace(
        preprocessor=types.SimpleNamespace(transform=lambda df: df),
        model=DummyPointModel(),
        meta={"model_name": "xgb", "version": "1"},
    )
    
    from forecastit.modeling.registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "load_champion", lambda *a, **k: (dummy.preprocessor, dummy.model, dummy.meta))

    pr = Predictor()
    out = pr.predict("S1", "I1", "2023-01-01", "2023-01-02")
    # fallback should compute intervals
    assert {"yhat_lower", "yhat_upper"}.issubset(out.columns)


def test_predictor_explain_prediction(monkeypatch):
    """Test predictor explain functionality."""
    dummy = types.SimpleNamespace(
        preprocessor=types.SimpleNamespace(transform=lambda df: df),
        model=DummyPointModel(),
        meta={"model_name": "xgb", "version": "1"},
    )
    
    from forecastit.modeling.registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "load_champion", lambda *a, **k: (dummy.preprocessor, dummy.model, dummy.meta))

    pr = Predictor()
    explanation = pr.explain_prediction("S1", "I1")
    
    # Should return explanation dict
    assert isinstance(explanation, dict)
    assert "store_id" in explanation
    assert "item_id" in explanation
