#!/usr/bin/env python3
"""Quick demo of ForecastIt functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demo_synthetic_data():
    """Demo synthetic data generation."""
    print("🔧 Generating synthetic data...")

    try:
        from forecastit.config.settings import Settings
        from forecastit.data.make_synthetic import generate_synthetic_data

        settings = Settings()

        # Generate small dataset
        data = generate_synthetic_data(
            start_date="2023-01-01",
            end_date="2023-01-31",
            stores=["store_01", "store_02"],
            items=["item_001", "item_002", "item_003"],
            seed=42,
            settings=settings,
        )

        print(f"✅ Generated data with shape: {data.shape}")
        print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"   Stores: {data['store_id'].unique()}")
        print(f"   Items: {data['item_id'].unique()}")
        print(f"   Sales range: ${data['sales'].min():.2f} - ${data['sales'].max():.2f}")

        return data

    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return None

def demo_feature_engineering(data):
    """Demo feature engineering."""
    print("\n🔧 Building features...")

    try:
        from forecastit.config.settings import Settings
        from forecastit.features.build import FeatureBuilder

        settings = Settings()
        feature_builder = FeatureBuilder(settings)

        # Build features
        features_data = feature_builder.build_features(data)

        print("✅ Features built successfully")
        print(f"   Original columns: {len(data.columns)}")
        print(f"   Feature columns: {len(features_data.columns)}")
        print(f"   Features added: {len(features_data.columns) - len(data.columns)}")

        # Show some feature names
        new_features = set(features_data.columns) - set(data.columns)
        print(f"   Sample features: {list(new_features)[:5]}")

        return features_data

    except Exception as e:
        print(f"❌ Error building features: {e}")
        return None

def demo_baseline_models(data):
    """Demo baseline model training."""
    print("\n🔧 Training baseline models...")

    try:
        from forecastit.modeling.baselines import BaselineModels

        baselines = BaselineModels()

        # Train all baseline models
        baseline_forecasts = baselines.generate_all_baselines(
            data,
            horizon=7,
        )

        print("✅ Baseline models trained successfully")
        print(f"   Models: {list(baseline_forecasts.keys())}")

        for model_name, forecasts in baseline_forecasts.items():
            print(f"   {model_name}: {len(forecasts)} forecasts generated")

        return baseline_forecasts

    except Exception as e:
        print(f"❌ Error training models: {e}")
        return None

def demo_api_schemas():
    """Demo API schema validation."""
    print("\n🔧 Testing API schemas...")

    try:

        from forecastit.api.schemas import ForecastRequest, InventoryRequest

        # Test forecast request
        forecast_req = ForecastRequest(
            store_id="store_01",
            item_id="item_001",
            start_date="2024-01-01",
            end_date="2024-01-07",
            promo_plan=[0, 1, 0, 1, 0, 0, 0],
        )

        # Test inventory request
        inventory_req = InventoryRequest(
            store_id="store_01",
            item_id="item_001",
            service_level=0.95,
            lead_time=7,
            holding_cost=0.1,
            stockout_cost=10.0,
        )

        print("✅ API schemas validated successfully")
        print(f"   Forecast request: {forecast_req.store_id} / {forecast_req.item_id}")
        print(f"   Inventory request: {inventory_req.service_level} service level")

        return True

    except Exception as e:
        print(f"❌ Error validating schemas: {e}")
        return False

def demo_inventory_calculations():
    """Demo inventory optimization calculations."""
    print("\n🔧 Testing inventory calculations...")

    try:
        from forecastit.config.settings import Settings
        from forecastit.inference.inventory import InventoryOptimizer

        settings = Settings()
        optimizer = InventoryOptimizer(settings)

        # Mock forecast data
        forecasts = [
            {"yhat": 100, "yhat_lower": 80, "yhat_upper": 120},
            {"yhat": 110, "yhat_lower": 90, "yhat_upper": 130},
            {"yhat": 105, "yhat_lower": 85, "yhat_upper": 125},
            {"yhat": 115, "yhat_lower": 95, "yhat_upper": 135},
            {"yhat": 120, "yhat_lower": 100, "yhat_upper": 140},
        ]

        # Calculate inventory metrics
        metrics = optimizer.calculate_inventory_metrics(
            forecasts=forecasts,
            service_level=0.95,
            lead_time=7,
            holding_cost=0.1,
            stockout_cost=10.0,
        )

        print("✅ Inventory calculations completed")
        print(f"   Expected demand: {metrics['expected_demand']:.1f}")
        print(f"   Safety stock: {metrics['safety_stock']:.1f}")
        print(f"   Reorder point: {metrics['reorder_point']:.1f}")
        print(f"   Stockout probability: {metrics['stockout_probability']:.1%}")

        return True

    except Exception as e:
        print(f"❌ Error in inventory calculations: {e}")
        return False

def main():
    """Run the quick demo."""
    print("🚀 ForecastIt Quick Demo")
    print("=" * 40)

    # Demo 1: Synthetic data generation
    data = demo_synthetic_data()
    if data is None:
        print("❌ Demo failed at data generation")
        return 1

    # Demo 2: Feature engineering
    features_data = demo_feature_engineering(data)
    if features_data is None:
        print("❌ Demo failed at feature engineering")
        return 1

    # Demo 3: Baseline models
    baseline_forecasts = demo_baseline_models(features_data)
    if baseline_forecasts is None:
        print("❌ Demo failed at model training")
        return 1

    # Demo 4: API schemas
    if not demo_api_schemas():
        print("❌ Demo failed at API schema validation")
        return 1

    # Demo 5: Inventory calculations
    if not demo_inventory_calculations():
        print("❌ Demo failed at inventory calculations")
        return 1

    print("\n" + "=" * 40)
    print("🎉 All demos completed successfully!")
    print("\n📝 System is ready for:")
    print("   • Data generation and cleaning")
    print("   • Feature engineering")
    print("   • Model training and evaluation")
    print("   • API inference")
    print("   • Inventory optimization")
    print("   • Dashboard visualization")

    return 0

if __name__ == "__main__":
    sys.exit(main())
