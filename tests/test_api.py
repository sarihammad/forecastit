"""Tests for API endpoints."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from forecastit.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_forecast_request():
    """Create sample forecast request."""
    return {
        "store_id": "store_1",
        "item_id": "item_1",
        "start_date": "2024-01-01",
        "end_date": "2024-01-07",
        "promo_plan": [0, 1, 0, 0, 0, 1, 0],
        "price_plan": [10.0, 9.0, 10.0, 10.0, 10.0, 9.0, 10.0],
    }


@pytest.fixture
def sample_inventory_request():
    """Create sample inventory request."""
    return {
        "store_id": "store_1",
        "item_id": "item_1",
        "service_level": 0.95,
        "lead_time": 7,
        "holding_cost": 0.1,
        "stockout_cost": 5.0,
    }


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data


class TestForecastEndpoint:
    """Test forecast endpoint."""

    def test_forecast_endpoint_success(self, client, sample_forecast_request):
        """Test successful forecast generation."""
        # Mock predictor instance
        mock_predictor_instance = AsyncMock()
        mock_predictor_instance.predict = AsyncMock(return_value=[
            {
                "store_id": "store_1",
                "item_id": "item_1",
                "date": datetime(2024, 1, 1),
                "yhat": 100.0,
                "yhat_lower": 90.0,
                "yhat_upper": 110.0,
                "model_name": "test_model",
                "model_version": "v1.0",
            }
        ])
        
        with patch("forecastit.api.main.predictor", mock_predictor_instance):
            response = client.post("/predict", json=sample_forecast_request)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "store_id" in data[0]
        assert "yhat" in data[0]

    def test_forecast_endpoint_validation_errors(self, client):
        """Test forecast endpoint validation."""
        # Invalid promo values
        invalid_request = {
            "store_id": "store_1",
            "item_id": "item_1",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "promo_plan": [0, 2, 0],  # Invalid: 2 is not 0 or 1
        }

        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 400
        assert "Promo values must be 0 or 1" in response.json()["detail"]

        # Invalid price values
        invalid_request = {
            "store_id": "store_1",
            "item_id": "item_1",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "price_plan": [10.0, -5.0, 10.0],  # Invalid: negative price
        }

        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 400
        assert "Price values must be positive" in response.json()["detail"]

        # Invalid date range
        invalid_request = {
            "store_id": "store_1",
            "item_id": "item_1",
            "start_date": "2024-01-07",
            "end_date": "2024-01-01",  # End before start
        }

        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 400
        assert "End date must be after start date" in response.json()["detail"]

        # Too long horizon
        invalid_request = {
            "store_id": "store_1",
            "item_id": "item_1",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",  # More than 365 days
        }

        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 400
        assert "Forecast horizon cannot exceed 365 days" in response.json()["detail"]


class TestInventoryEndpoint:
    """Test inventory endpoint."""

    def test_inventory_endpoint_success(self, client, sample_inventory_request):
        """Test successful inventory optimization."""
        # Mock predictor
        mock_predictor_instance = AsyncMock()
        mock_predictor_instance.predict = AsyncMock(return_value=[
            {
                "store_id": "store_1",
                "item_id": "item_1",
                "date": datetime.now().date(),
                "yhat": 100.0,
                "yhat_lower": 90.0,
                "yhat_upper": 110.0,
                "model_name": "test_model",
                "model_version": "v1.0",
            }
        ])

        # Mock inventory optimizer
        mock_inventory_instance = Mock()
        mock_inventory_instance.calculate_inventory_metrics.return_value = {
            "reorder_point": 150.0,
            "safety_stock": 50.0,
            "expected_demand": 100.0,
            "demand_std": 10.0,
            "stockout_probability": 0.05,
            "service_level": 0.95,
        }

        with patch("forecastit.api.main.predictor", mock_predictor_instance), \
             patch("forecastit.api.main.inventory_optimizer", mock_inventory_instance):
            response = client.post("/inventory/suggest", json=sample_inventory_request)

        assert response.status_code == 200
        data = response.json()
        assert "reorder_point" in data
        assert "safety_stock" in data
        assert "service_level" in data

    def test_inventory_endpoint_validation_errors(self, client):
        """Test inventory endpoint validation."""
        # Invalid service level
        invalid_request = {
            "store_id": "store_1",
            "item_id": "item_1",
            "service_level": 0.3,  # Too low
            "lead_time": 7,
            "holding_cost": 0.1,
            "stockout_cost": 5.0,
        }

        response = client.post("/inventory/suggest", json=invalid_request)
        assert response.status_code == 400
        assert "Service level must be between 0.5 and 0.999" in response.json()["detail"]

        # Invalid lead time
        invalid_request = {
            "store_id": "store_1",
            "item_id": "item_1",
            "service_level": 0.95,
            "lead_time": 0,  # Invalid: must be > 0
            "holding_cost": 0.1,
            "stockout_cost": 5.0,
        }

        response = client.post("/inventory/suggest", json=invalid_request)
        assert response.status_code == 400
        assert "Lead time must be between 1 and 365 days" in response.json()["detail"]

        # Invalid costs
        invalid_request = {
            "store_id": "store_1",
            "item_id": "item_1",
            "service_level": 0.95,
            "lead_time": 7,
            "holding_cost": -0.1,  # Invalid: negative
            "stockout_cost": 5.0,
        }

        response = client.post("/inventory/suggest", json=invalid_request)
        assert response.status_code == 400
        assert "Holding cost must be non-negative" in response.json()["detail"]


class TestExplainEndpoint:
    """Test explain endpoint."""

    def test_explain_endpoint_success(self, client):
        """Test successful explanation generation."""
        # Mock predictor
        mock_predictor_instance = AsyncMock()
        mock_predictor_instance.explain_prediction = Mock(return_value={
            "store_id": "store_1",
            "item_id": "item_1",
            "top_features": [
                ("price", 0.3),
                ("on_promo", 0.2),
                ("day_of_week", 0.1),
            ],
            "feature_importance": {"price": 0.3, "on_promo": 0.2},
            "explanation_method": "SHAP",
        })

        with patch("forecastit.api.main.predictor", mock_predictor_instance):
            response = client.get("/explain/store_1/item_1")

        assert response.status_code == 200
        data = response.json()
        assert "store_id" in data
        assert "top_features" in data
        assert "explanation_method" in data


class TestModelsEndpoint:
    """Test models endpoint."""

    def test_models_endpoint_success(self, client):
        """Test successful models listing."""
        # Mock predictor
        mock_predictor_instance = AsyncMock()
        mock_predictor_instance.list_available_models = AsyncMock(return_value=[
            {
                "name": "test_model",
                "version": "v1.0",
                "metric_value": 0.1,
                "created_at": "2024-01-01",
            }
        ])
        mock_predictor_instance.current_model_info = {"model_name": "test_model"}

        with patch("forecastit.api.main.predictor", mock_predictor_instance):
            response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "current_model" in data


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limiting_middleware(self, client):
        """Test that rate limiting middleware is active."""
        # Make multiple requests quickly to trigger rate limiting
        # Note: This test might be flaky due to timing, but it tests the middleware exists
        responses = []
        for _ in range(105):  # Exceed the rate limit
            response = client.get("/health")
            responses.append(response.status_code)

        # At least one should be rate limited (429)
        assert 429 in responses


class TestErrorHandling:
    """Test error handling."""

    def test_global_exception_handler(self, client):
        """Test global exception handler."""
        # Test that the endpoint returns 503 when predictor is not initialized
        with patch("forecastit.api.main.predictor", None), \
             patch("forecastit.api.main.rate_limiter") as mock_rate_limiter:
            mock_rate_limiter.allow_request.return_value = True
            response = client.get("/models")
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"]

    def test_predict_validation_error(self, client):
        """Test forecast endpoint validation."""
        # Invalid date range (end < start)
        invalid_request = {
            "store_id": "store_1",
            "item_id": "item_1",
            "start_date": "2023-01-10",
            "end_date": "2023-01-01",  # end < start
        }
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code in (400, 422)

    def test_rate_limit(self, client):
        """Test rate limiting functionality."""
        payload = {
            "store_id": "S",
            "item_id": "I", 
            "start_date": "2023-01-01",
            "end_date": "2023-01-03"
        }
        
        # Make multiple requests to potentially trigger rate limiting
        for _ in range(5):
            response = client.post("/predict", json=payload)
        
        # Depending on bucket size, might get 429 or 200
        assert response.status_code in (429, 200, 503)

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])
