"""Tests for inventory optimization functionality."""

import pytest
import numpy as np
from forecastit.inference.inventory import InventoryOptimizer


def test_inventory_optimizer_basic():
    """Test basic inventory optimization calculations."""
    from forecastit.config.settings import Settings
    optimizer = InventoryOptimizer(Settings())
    
    # Test basic ROP calculation
    expected_demand = 100
    lead_time = 7
    service_level = 0.95
    demand_std = 10
    
    rop = optimizer.calculate_optimal_reorder_point(
        expected_demand=expected_demand,
        lead_time=lead_time,
        service_level=service_level,
        demand_std=demand_std
    )
    
    assert rop > 0
    assert rop >= expected_demand * lead_time  # Should be at least expected demand during lead time


def test_inventory_optimizer_edge_cases():
    """Test inventory optimization edge cases."""
    from forecastit.config.settings import Settings
    optimizer = InventoryOptimizer(Settings())
    
    # Test with zero demand
    rop_zero = optimizer.calculate_reorder_point(
        expected_demand=0,
        lead_time=7,
        service_level=0.95,
        demand_std=0
    )
    assert rop_zero >= 0
    
    # Test with very high service level
    rop_high_sl = optimizer.calculate_reorder_point(
        expected_demand=100,
        lead_time=7,
        service_level=0.999,
        demand_std=10
    )
    assert rop_high_sl > 0
    
    # Test with very low service level
    rop_low_sl = optimizer.calculate_reorder_point(
        expected_demand=100,
        lead_time=7,
        service_level=0.5,
        demand_std=10
    )
    assert rop_low_sl >= 0


def test_inventory_metrics_calculation():
    """Test comprehensive inventory metrics calculation."""
    from forecastit.config.settings import Settings
    optimizer = InventoryOptimizer(Settings())
    
    # Mock forecast data
    forecasts = [
        {"yhat": 100, "yhat_lower": 90, "yhat_upper": 110},
        {"yhat": 105, "yhat_lower": 95, "yhat_upper": 115},
        {"yhat": 98, "yhat_lower": 88, "yhat_upper": 108},
    ]
    
    metrics = optimizer.calculate_inventory_metrics(
        forecasts=forecasts,
        service_level=0.95,
        lead_time=7,
        holding_cost=0.1,
        stockout_cost=5.0
    )
    
    assert "reorder_point" in metrics
    assert "safety_stock" in metrics
    assert "expected_demand" in metrics
    assert "demand_std" in metrics
    assert "stockout_probability" in metrics
    assert "service_level" in metrics
    
    assert metrics["service_level"] == 0.95
    assert metrics["expected_demand"] > 0
    assert metrics["reorder_point"] > 0
    assert metrics["safety_stock"] >= 0


def test_inventory_optimizer_invalid_inputs():
    """Test inventory optimizer with invalid inputs."""
    from forecastit.config.settings import Settings
    optimizer = InventoryOptimizer(Settings())
    
    # Test with negative values
    with pytest.raises(ValueError):
        optimizer.calculate_reorder_point(
            expected_demand=-1,
            lead_time=7,
            service_level=0.95,
            demand_std=10
        )
    
    # Test with invalid service level
    with pytest.raises(ValueError):
        optimizer.calculate_reorder_point(
            expected_demand=100,
            lead_time=7,
            service_level=1.5,  # > 1
            demand_std=10
        )
    
    with pytest.raises(ValueError):
        optimizer.calculate_reorder_point(
            expected_demand=100,
            lead_time=7,
            service_level=-0.1,  # < 0
            demand_std=10
        )
