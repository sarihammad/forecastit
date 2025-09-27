"""Prefect training workflow for ForecastIt."""

from typing import Any

import pandas as pd
import structlog
from prefect import flow, task

from forecastit.cleaning.clean import DataCleaner
from forecastit.config.settings import Settings
from forecastit.data.loaders import DataLoader
from forecastit.features.build import FeatureBuilder
from forecastit.modeling.backtest import RollingBacktester
from forecastit.modeling.classical import ProphetForecaster, SarimaxForecaster
from forecastit.modeling.ml_models import (
    LGBMQuantileForecaster,
    LGBMRegressorForecaster,
    XGBRegressorForecaster,
)
from forecastit.modeling.registry import ModelRegistry
from forecastit.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


@task(name="load_data")
def load_data_task(settings: Settings) -> pd.DataFrame:
    """Load and validate data."""
    logger.info("Loading data")

    loader = DataLoader(settings)
    data = loader.load_synthetic()

    logger.info("Data loaded successfully", shape=data.shape)
    return data


@task(name="clean_data")
def clean_data_task(data: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Clean and preprocess data."""
    logger.info("Cleaning data")

    cleaner = DataCleaner(settings)
    cleaned_data = cleaner.clean_data(data)

    logger.info("Data cleaned successfully", shape=cleaned_data.shape)
    return cleaned_data


@task(name="build_features")
def build_features_task(data: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Build features for modeling."""
    logger.info("Building features")

    feature_builder = FeatureBuilder(settings)
    features_data = feature_builder.build_features(data)

    logger.info("Features built successfully", shape=features_data.shape)
    return features_data


@task(name="run_backtesting")
def run_backtesting_task(
    data: pd.DataFrame,
    feature_builder: FeatureBuilder,
    settings: Settings,
) -> dict[str, Any]:
    """Run rolling backtesting for all model families."""
    logger.info("Starting rolling backtesting")

    # Initialize backtester
    backtester = RollingBacktester(settings)

    # Define feature function
    def feature_fn(df: pd.DataFrame) -> pd.DataFrame:
        return feature_builder.build_features(df)

    # Define model families and their fit/predict functions
    model_families = {
        "naive": {
            "fit_fn": lambda train_features, y: None,  # No fitting needed
            "predict_fn": lambda model, val_features: val_features["sales"].shift(1).fillna(val_features["sales"].mean()),
        },
        "sarimax": {
            "fit_fn": lambda train_features, y: SarimaxForecaster().fit(train_features),
            "predict_fn": lambda model, val_features: model.predict(len(val_features)),
        },
        "lgbm_point": {
            "fit_fn": lambda train_features, y: LGBMRegressorForecaster().fit(train_features, y),
            "predict_fn": lambda model, val_features: model.predict(val_features),
        },
        "lgbm_quantile": {
            "fit_fn": lambda train_features, y: LGBMQuantileForecaster().fit(train_features, y),
            "predict_fn": lambda model, val_features: model.predict(val_features),
        },
        "xgb": {
            "fit_fn": lambda train_features, y: XGBRegressorForecaster().fit(train_features, y),
            "predict_fn": lambda model, val_features: model.predict(val_features),
        },
    }
    
    # Conditionally add Prophet if enabled and not disabled
    if settings.enable_prophet and not settings.disable_prophet:
        try:
            import prophet
            model_families["prophet"] = {
                "fit_fn": lambda train_features, y: ProphetForecaster().fit(train_features),
                "predict_fn": lambda model, val_features: model.predict(len(val_features)),
            }
            logger.info("Prophet enabled - cmdstan available")
        except ImportError as e:
            logger.warning(f"Prophet disabled - cmdstan not available: {e}")
    else:
        logger.info("Prophet disabled via configuration")

    # Run backtesting for each model family
    backtest_results = {}
    metrics = ["RMSE", "MAPE", "sMAPE", "MASE"]

    for model_name, model_functions in model_families.items():
        try:
            logger.info(f"Backtesting {model_name}")

            result = backtester.rolling_backtest(
                df=data,
                series_cols=["store_id", "item_id"],
                target_col="sales",
                date_col="date",
                horizon=settings.default_horizon,
                n_folds=settings.cv_folds,
                fit_fn=model_functions["fit_fn"],
                predict_fn=model_functions["predict_fn"],
                feature_fn=feature_fn,
                metrics=metrics,
                model_name=model_name,
            )

            backtest_results[model_name] = result

        except Exception as e:
            logger.error(f"Failed to backtest {model_name}: {e}")
            continue

    logger.info("Backtesting completed", models=list(backtest_results.keys()))
    return backtest_results


@task(name="select_champion")
def select_champion_task(
    backtest_results: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Select champion model based on backtesting results."""
    logger.info("Selecting champion model")

    # Calculate weighted scores (volume-weighted sMAPE)
    model_scores = {}

    for model_name, results in backtest_results.items():
        aggregated_metrics = results.get("aggregated_metrics", {})

        # Use sMAPE as primary metric (lower is better)
        primary_metric = aggregated_metrics.get("sMAPE", float("inf"))

        # Add penalty for high variance
        variance_penalty = aggregated_metrics.get("sMAPE_std", 0) * 0.1

        model_scores[model_name] = primary_metric + variance_penalty

    # Select best model
    champion_model = min(model_scores.items(), key=lambda x: x[1])
    champion_name = champion_model[0]
    champion_score = champion_model[1]

    logger.info(
        "Champion model selected",
        model=champion_name,
        score=champion_score,
        all_scores=model_scores,
    )

    return {
        "champion_model": champion_name,
        "champion_score": champion_score,
        "all_scores": model_scores,
        "champion_results": backtest_results[champion_name],
    }


@task(name="register_champion_model")
def register_champion_model_task(
    champion_selection: dict[str, Any],
    feature_builder: FeatureBuilder,
    settings: Settings,
) -> str:
    """Register the champion model to MLflow."""
    logger.info("Registering champion model")

    champion_name = champion_selection["champion_model"]
    champion_score = champion_selection["champion_score"]
    champion_results = champion_selection["champion_results"]

    logger.info(
        "Registering champion model",
        model=champion_name,
        score=champion_score,
    )

    # Initialize registry
    registry = ModelRegistry(settings)

    # Create model signature for inference
    from mlflow.types.schema import ColSpec, Schema

    # Define input schema
    input_schema = Schema([
        ColSpec("string", "store_id"),
        ColSpec("string", "item_id"),
        ColSpec("datetime", "date"),
        ColSpec("double", "sales"),
        ColSpec("double", "price", optional=True),
        ColSpec("boolean", "on_promo", optional=True),
    ])

    # Define output schema
    output_schema = Schema([
        ColSpec("double", "yhat"),
        ColSpec("double", "yhat_lower", optional=True),
        ColSpec("double", "yhat_upper", optional=True),
    ])

    from mlflow.models.signature import ModelSignature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Register model with metadata
    model_metadata = {
        "champion_model": champion_name,
        "champion_score": champion_score,
        "backtest_metrics": champion_results.get("aggregated_metrics", {}),
        "successful_folds": champion_results.get("successful_folds", 0),
        "total_predictions": champion_results.get("total_predictions", 0),
        "feature_columns": feature_builder.get_feature_columns(),
    }

    # Create a mock model URI (in practice, this would be the actual model artifact)
    model_uri = "models://forecastit_champion/1"

    # Register model
    version = registry.register_model(
        model_uri=model_uri,
        model_name="forecastit_champion",
        description=f"Champion model: {champion_name} with score {champion_score:.4f}",
        stage="Production",
        signature=signature,
        metadata=model_metadata,
    )

    logger.info("Champion model registered", version=version)
    return version


@flow(name="forecastit_training_flow")
def run_training_flow(
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Main training workflow with comprehensive backtesting and model selection."""
    if settings is None:
        settings = Settings()

    setup_logging(settings)
    logger.info("Starting comprehensive training flow")

    try:
        # Load data
        data = load_data_task(settings)

        # Clean data
        cleaned_data = clean_data_task(data, settings)

        # Initialize feature builder
        feature_builder = FeatureBuilder(settings)

        # Build features
        features_data = build_features_task(cleaned_data, settings)

        # Run comprehensive backtesting
        backtest_results = run_backtesting_task(features_data, feature_builder, settings)

        # Select champion model
        champion_selection = select_champion_task(backtest_results, settings)

        # Register champion model
        model_version = register_champion_model_task(champion_selection, feature_builder, settings)

        result = {
            "status": "success",
            "data_shape": features_data.shape,
            "models_backtested": len(backtest_results),
            "champion_model": champion_selection["champion_model"],
            "champion_score": champion_selection["champion_score"],
            "all_scores": champion_selection["all_scores"],
            "model_version": model_version,
            "feature_columns": feature_builder.get_feature_columns(),
        }

        logger.info("Training flow completed successfully", result=result)
        return result

    except Exception as e:
        logger.error("Training flow failed", error=str(e))
        raise


if __name__ == "__main__":
    # Run the training flow
    result = run_training_flow()
    print(f"Training completed: {result}")
