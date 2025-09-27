#!/usr/bin/env python3
"""Test script to validate ForecastIt system."""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from forecastit.config.settings import Settings
        print("✅ Settings imported successfully")

        from forecastit.data.make_synthetic import generate_synthetic_data
        print("✅ Synthetic data generation imported successfully")

        from forecastit.data.loaders import DataLoader
        print("✅ Data loaders imported successfully")

        from forecastit.cleaning.clean import DataCleaner
        print("✅ Data cleaning imported successfully")

        from forecastit.features.build import FeatureBuilder
        print("✅ Feature building imported successfully")

        from forecastit.modeling.datasets import TimeSeriesDataset
        print("✅ Time series dataset imported successfully")

        from forecastit.modeling.metrics import MetricsCalculator
        print("✅ Metrics calculator imported successfully")

        from forecastit.modeling.baselines import BaselineModels
        print("✅ Baseline models imported successfully")

        from forecastit.modeling.registry import ModelRegistry
        print("✅ Model registry imported successfully")

        from forecastit.inference.predictor import Predictor
        print("✅ Predictor imported successfully")

        from forecastit.inference.inventory import InventoryOptimizer
        print("✅ Inventory optimizer imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data generation."""
    print("\nTesting synthetic data generation...")

    try:
        from forecastit.config.settings import Settings
        from forecastit.data.make_synthetic import generate_synthetic_data

        settings = Settings()

        # Generate small dataset
        data = generate_synthetic_data(
            start_date="2023-01-01",
            end_date="2023-01-07",
            stores=["store_01", "store_02"],
            items=["item_001", "item_002"],
            seed=42,
            settings=settings,
        )

        print(f"✅ Generated synthetic data with shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"   Unique stores: {data['store_id'].nunique()}")
        print(f"   Unique items: {data['item_id'].nunique()}")

        return True

    except Exception as e:
        print(f"❌ Synthetic data generation failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering pipeline."""
    print("\nTesting feature engineering...")

    try:
        from forecastit.config.settings import Settings
        from forecastit.data.make_synthetic import generate_synthetic_data
        from forecastit.features.build import FeatureBuilder

        settings = Settings()

        # Generate data
        data = generate_synthetic_data(
            start_date="2023-01-01",
            end_date="2023-01-30",
            stores=["store_01"],
            items=["item_001"],
            seed=42,
            settings=settings,
        )

        # Build features
        feature_builder = FeatureBuilder(settings)
        features_data = feature_builder.build_features(data)

        print("✅ Feature engineering completed")
        print(f"   Original columns: {len(data.columns)}")
        print(f"   Feature columns: {len(features_data.columns)}")
        print(f"   Features added: {len(features_data.columns) - len(data.columns)}")

        # Check specific features
        calendar_features = [col for col in features_data.columns if col.startswith(("day_", "month", "quarter", "is_"))]
        lag_features = [col for col in features_data.columns if "lag" in col or "rolling" in col]

        print(f"   Calendar features: {len(calendar_features)}")
        print(f"   Lag/rolling features: {len(lag_features)}")

        return True

    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False

def test_baseline_models():
    """Test baseline model training."""
    print("\nTesting baseline models...")

    try:
        from forecastit.config.settings import Settings
        from forecastit.data.make_synthetic import generate_synthetic_data
        from forecastit.features.build import FeatureBuilder
        from forecastit.modeling.baselines import BaselineModels

        settings = Settings()

        # Generate and process data
        data = generate_synthetic_data(
            start_date="2023-01-01",
            end_date="2023-01-30",
            stores=["store_01"],
            items=["item_001"],
            seed=42,
            settings=settings,
        )

        feature_builder = FeatureBuilder(settings)
        features_data = feature_builder.build_features(data)

        # Train baseline models
        baselines = BaselineModels()
        baseline_forecasts = baselines.generate_all_baselines(
            features_data,
            horizon=7,
        )

        print("✅ Baseline models trained successfully")
        print(f"   Models trained: {list(baseline_forecasts.keys())}")

        for model_name, forecasts in baseline_forecasts.items():
            print(f"   {model_name}: {len(forecasts)} forecasts")

        return True

    except Exception as e:
        print(f"❌ Baseline model training failed: {e}")
        return False

def test_api_schemas():
    """Test API schema validation."""
    print("\nTesting API schemas...")

    try:

        from forecastit.api.schemas import (
            ForecastRequest,
            InventoryRequest,
        )

        # Test forecast request
        ForecastRequest(
            store_id="store_01",
            item_id="item_001",
            start_date="2024-01-01",
            end_date="2024-01-07",
        )
        print("✅ Forecast request schema validated")

        # Test inventory request
        InventoryRequest(
            store_id="store_01",
            item_id="item_001",
            service_level=0.95,
            lead_time=7,
        )
        print("✅ Inventory request schema validated")

        return True

    except Exception as e:
        print(f"❌ API schema validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ForecastIt System Validation")
    print("=" * 50)

    tests = [
        test_imports,
        test_synthetic_data,
        test_feature_engineering,
        test_baseline_models,
        test_api_schemas,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
