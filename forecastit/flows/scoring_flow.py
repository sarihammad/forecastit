"""Prefect scoring workflow for ForecastIt."""


import pandas as pd
import structlog
import numpy as np
from prefect import flow, task

from forecastit.config.settings import Settings
from forecastit.data.loaders import DataLoader
from forecastit.inference.predictor import Predictor
from forecastit.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two distributions.
    
    Args:
        expected: Expected/reference distribution.
        actual: Actual/current distribution.
        bins: Number of bins for histogram.
        
    Returns:
        PSI score (0-0.1: no change, 0.1-0.2: moderate change, >0.2: significant change).
    """
    try:
        # Remove NaN values
        expected = expected.dropna()
        actual = actual.dropna()
        
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        # Create bins based on expected distribution
        if expected.dtype in ['object', 'category']:
            # For categorical data, use value counts
            expected_counts = expected.value_counts(normalize=True)
            actual_counts = actual.value_counts(normalize=True)
            
            # Get all unique values
            all_values = set(expected_counts.index) | set(actual_counts.index)
            
            psi = 0.0
            for value in all_values:
                expected_pct = expected_counts.get(value, 0.0)
                actual_pct = actual_counts.get(value, 0.0)
                
                # Avoid division by zero and log of zero
                if expected_pct > 0 and actual_pct > 0:
                    psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        else:
            # For numerical data, create histogram bins
            # Use quantile-based bins for better distribution
            quantiles = np.linspace(0, 1, bins + 1)
            expected_bins = np.quantile(expected, quantiles)
            
            # Ensure unique bin edges
            expected_bins = np.unique(expected_bins)
            
            # Calculate histogram for expected distribution
            expected_hist, _ = np.histogram(expected, bins=expected_bins)
            expected_pct = expected_hist / len(expected)
            
            # Calculate histogram for actual distribution
            actual_hist, _ = np.histogram(actual, bins=expected_bins)
            actual_pct = actual_hist / len(actual)
            
            # Calculate PSI
            psi = 0.0
            for i in range(len(expected_pct)):
                if expected_pct[i] > 0 and actual_pct[i] > 0:
                    psi += (actual_pct[i] - expected_pct[i]) * np.log(actual_pct[i] / expected_pct[i])
        
        return max(0.0, psi)  # Ensure non-negative
        
    except Exception as e:
        logger.error(f"PSI calculation failed: {e}")
        return 0.0


@task(name="load_new_data")
def load_new_data_task(settings: Settings) -> pd.DataFrame:
    """Load new data for scoring."""
    logger.info("Loading new data for scoring")

    # In practice, this would load from your data source
    # For now, we'll use synthetic data
    loader = DataLoader(settings)
    data = loader.load_synthetic()

    # Filter to recent data (last 7 days)
    recent_data = data[data["date"] >= data["date"].max() - pd.Timedelta(days=7)]

    logger.info("New data loaded", shape=recent_data.shape)
    return recent_data


@task(name="generate_predictions")
def generate_predictions_task(
    data: pd.DataFrame,
    settings: Settings,
) -> list[dict]:
    """Generate predictions for new data."""
    logger.info("Generating predictions")

    predictor = Predictor(settings)

    # Get unique store-item combinations
    unique_combinations = data[["store_id", "item_id"]].drop_duplicates()

    predictions = []

    for _, row in unique_combinations.iterrows():
        store_id = row["store_id"]
        item_id = row["item_id"]

        # Generate forecast for next 7 days
        from datetime import date, timedelta
        start_date = date.today()
        end_date = start_date + timedelta(days=7)

        try:
            forecast = predictor.predict(
                store_id=store_id,
                item_id=item_id,
                start_date=start_date,
                end_date=end_date,
            )

            predictions.extend(forecast)

        except Exception as e:
            logger.warning(
                "Failed to generate prediction",
                store_id=store_id,
                item_id=item_id,
                error=str(e),
            )

    logger.info("Predictions generated", count=len(predictions))
    return predictions


@task(name="check_data_drift")
def check_data_drift_task(
    new_data: pd.DataFrame,
    historical_data: pd.DataFrame,
    settings: Settings,
) -> dict[str, float]:
    """Check for data drift using PSI and statistical measures."""
    logger.info("Checking for data drift")

    drift_metrics = {}

    # Key features to monitor for drift
    features_to_check = ["sales", "price", "on_promo"]
    
    for feature in features_to_check:
        if feature in new_data.columns and feature in historical_data.columns:
            # Calculate PSI (Population Stability Index)
            psi_score = calculate_psi(new_data[feature], historical_data[feature])
            drift_metrics[f"{feature}_psi"] = psi_score
            
            # Statistical drift measures
            new_mean = new_data[feature].mean()
            historical_mean = historical_data[feature].mean()
            new_std = new_data[feature].std()
            historical_std = historical_data[feature].std()

            # Mean and variance drift
            mean_drift = abs(new_mean - historical_mean) / historical_mean if historical_mean > 0 else 0
            std_drift = abs(new_std - historical_std) / historical_std if historical_std > 0 else 0

            drift_metrics[f"{feature}_mean_drift"] = mean_drift
            drift_metrics[f"{feature}_std_drift"] = std_drift
            
            # Log warnings for significant drift
            if psi_score > 0.2:  # PSI > 0.2 indicates significant drift
                logger.warning(
                    f"Significant drift detected in {feature}",
                    psi_score=psi_score,
                    feature=feature,
                )
            elif psi_score > 0.1:  # PSI > 0.1 indicates moderate drift
                logger.info(
                    f"Moderate drift detected in {feature}",
                    psi_score=psi_score,
                    feature=feature,
                )

    # Check for missing data patterns
    for col in new_data.columns:
        if col in historical_data.columns:
            new_missing = new_data[col].isnull().sum() / len(new_data)
            historical_missing = historical_data[col].isnull().sum() / len(historical_data)
            missing_drift = abs(new_missing - historical_missing)
            
            drift_metrics[f"{col}_missing_drift"] = missing_drift
            
            # Warn about significant missing data changes
            if missing_drift > 0.1:  # 10% change in missing data
                logger.warning(
                    f"Significant change in missing data for {col}",
                    missing_drift=missing_drift,
                    column=col,
                )

    logger.info("Data drift check completed", metrics=drift_metrics)
    return drift_metrics


@task(name="check_model_drift")
def check_model_drift_task(
    predictions: list[dict],
    historical_predictions: list[dict],
    settings: Settings,
) -> dict[str, float]:
    """Check for model drift."""
    logger.info("Checking for model drift")

    drift_metrics = {}

    if not predictions or not historical_predictions:
        logger.warning("No predictions available for drift check")
        return drift_metrics

    # Convert to DataFrames
    pred_df = pd.DataFrame(predictions)
    hist_pred_df = pd.DataFrame(historical_predictions)

    # Compare prediction statistics
    if "yhat" in pred_df.columns and "yhat" in hist_pred_df.columns:
        pred_mean = pred_df["yhat"].mean()
        hist_pred_mean = hist_pred_df["yhat"].mean()

        pred_std = pred_df["yhat"].std()
        hist_pred_std = hist_pred_df["yhat"].std()

        mean_drift = abs(pred_mean - hist_pred_mean) / hist_pred_mean if hist_pred_mean > 0 else 0
        std_drift = abs(pred_std - hist_pred_std) / hist_pred_std if hist_pred_std > 0 else 0

        drift_metrics["prediction_mean_drift"] = mean_drift
        drift_metrics["prediction_std_drift"] = std_drift

    logger.info("Model drift check completed", metrics=drift_metrics)
    return drift_metrics


@task(name="send_alerts")
def send_alerts_task(
    data_drift: dict[str, float],
    model_drift: dict[str, float],
    settings: Settings,
) -> bool:
    """Send alerts if drift is detected."""
    logger.info("Checking for alerts")

    # Define drift thresholds
    drift_threshold = 0.2  # 20% drift threshold

    alerts_sent = False

    # Check data drift
    for metric, value in data_drift.items():
        if value > drift_threshold:
            logger.warning(
                "Data drift detected",
                metric=metric,
                value=value,
                threshold=drift_threshold,
            )
            alerts_sent = True

    # Check model drift
    for metric, value in model_drift.items():
        if value > drift_threshold:
            logger.warning(
                "Model drift detected",
                metric=metric,
                value=value,
                threshold=drift_threshold,
            )
            alerts_sent = True

    if alerts_sent:
        # In practice, this would send actual alerts (email, Slack, etc.)
        logger.info("Drift alerts sent")
    else:
        logger.info("No drift alerts needed")

    return alerts_sent


@flow(name="forecastit_scoring_flow")
def run_scoring_flow(
    settings: Settings | None = None,
) -> dict[str, any]:
    """Main scoring workflow."""
    if settings is None:
        settings = Settings()

    setup_logging(settings)
    logger.info("Starting scoring flow")

    try:
        # Load new data
        new_data = load_new_data_task(settings)

        # Load historical data for comparison
        loader = DataLoader(settings)
        historical_data = loader.load_synthetic()

        # Generate predictions
        predictions = generate_predictions_task(new_data, settings)

        # Check data drift
        data_drift = check_data_drift_task(new_data, historical_data, settings)

        # Check model drift (simplified - would need historical predictions)
        model_drift = check_model_drift_task(predictions, [], settings)

        # Send alerts if needed
        alerts_sent = send_alerts_task(data_drift, model_drift, settings)

        result = {
            "status": "success",
            "new_data_shape": new_data.shape,
            "predictions_generated": len(predictions),
            "data_drift_detected": any(v > 0.2 for v in data_drift.values()),
            "model_drift_detected": any(v > 0.2 for v in model_drift.values()),
            "alerts_sent": alerts_sent,
        }

        logger.info("Scoring flow completed successfully", result=result)
        return result

    except Exception as e:
        logger.error("Scoring flow failed", error=str(e))
        raise


if __name__ == "__main__":
    # Run the scoring flow
    result = run_scoring_flow()
    print(f"Scoring completed: {result}")
