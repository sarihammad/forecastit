"""Synthetic data generation for testing and development."""

import random
from datetime import date

import numpy as np
import pandas as pd
import structlog

from forecastit.config.settings import Settings

logger = structlog.get_logger(__name__)


def generate_synthetic_data(
    start_date: date = date(2022, 1, 1),
    end_date: date = date(2023, 12, 31),
    stores: list[str] | None = None,
    items: list[str] | None = None,
    seed: int = 42,
    settings: Settings = None,
) -> pd.DataFrame:
    """Generate synthetic sales data with realistic patterns.

    Args:
        start_date: Start date for data generation.
        end_date: End date for data generation.
        stores: List of store IDs (generates default if None).
        items: List of item IDs (generates default if None).
        seed: Random seed for reproducibility.
        settings: Application settings.

    Returns:
        DataFrame with synthetic sales data.
    """
    if settings is None:
        settings = Settings()

    # Set random seed
    np.random.seed(seed)
    random.seed(seed)

    # Default stores and items
    if stores is None:
        stores = [f"store_{i:02d}" for i in range(1, 6)]
    if items is None:
        items = [f"item_{i:03d}" for i in range(1, 21)]

    logger.info(
        "Generating synthetic data",
        start_date=start_date,
        end_date=end_date,
        num_stores=len(stores),
        num_items=len(items),
        seed=seed,
    )

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate base data
    data = []

    for store in stores:
        for item in items:
            # Generate series-specific parameters
            store_item_seed = hash(f"{store}_{item}") % 2**32
            np.random.seed(store_item_seed)

            # Base sales level (varies by store and item)
            base_sales = np.random.lognormal(mean=3.0, sigma=1.0)

            # Seasonality parameters
            seasonal_amplitude = np.random.uniform(0.1, 0.5)
            seasonal_phase = np.random.uniform(0, 2 * np.pi)

            # Trend parameters
            trend_slope = np.random.normal(0, 0.001)

            # Generate time series
            series_data = []

            for i, dt in enumerate(date_range):
                # Base level
                sales = base_sales

                # Trend
                sales += trend_slope * i

                # Seasonality (annual)
                day_of_year = dt.timetuple().tm_yday
                seasonal_effect = seasonal_amplitude * np.sin(
                    2 * np.pi * day_of_year / 365.25 + seasonal_phase
                )
                sales *= (1 + seasonal_effect)

                # Weekly seasonality
                day_of_week = dt.weekday()
                weekly_effect = 0.2 * np.sin(2 * np.pi * day_of_week / 7)
                sales *= (1 + weekly_effect)

                # Holiday effects
                holiday_multiplier = _get_holiday_multiplier(dt)
                sales *= holiday_multiplier

                # Random noise
                noise = np.random.lognormal(mean=0, sigma=0.1)
                sales *= noise

                # Ensure non-negative
                sales = max(0, sales)

                series_data.append({
                    "date": dt.date(),
                    "store_id": store,
                    "item_id": item,
                    "sales": sales,
                })

            data.extend(series_data)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Add promotion and price features
    df = _add_promo_price_features(df, seed)

    # Add weather features (optional)
    df = _add_weather_features(df, seed)

    # Add holiday features
    df = _add_holiday_features(df)

    # Sort by store, item, date
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    logger.info(
        "Generated synthetic data",
        shape=df.shape,
        date_range=f"{df['date'].min()} to {df['date'].max()}",
        stores=list(df["store_id"].unique()),
        items=list(df["item_id"].unique()),
    )

    return df


def _get_holiday_multiplier(dt: pd.Timestamp) -> float:
    """Get holiday multiplier for a date.

    Args:
        dt: Date to check.

    Returns:
        Holiday multiplier (1.0 = no effect).
    """
    # Simple holiday effects
    month = dt.month
    day = dt.day

    # New Year's Day
    if month == 1 and day == 1:
        return 0.3

    # Valentine's Day
    if month == 2 and day == 14:
        return 1.5

    # Easter (approximate)
    if month == 4 and 10 <= day <= 20:
        return 1.2

    # Mother's Day (second Sunday in May)
    if month == 5 and 8 <= day <= 14 and dt.weekday() == 6:
        return 1.3

    # Father's Day (third Sunday in June)
    if month == 6 and 15 <= day <= 21 and dt.weekday() == 6:
        return 1.2

    # Independence Day
    if month == 7 and day == 4:
        return 0.4

    # Halloween
    if month == 10 and day == 31:
        return 1.4

    # Thanksgiving (fourth Thursday in November)
    if month == 11 and 22 <= day <= 28 and dt.weekday() == 3:
        return 0.2

    # Black Friday (day after Thanksgiving)
    if month == 11 and 23 <= day <= 29 and dt.weekday() == 4:
        return 2.0

    # Cyber Monday (Monday after Thanksgiving)
    if month == 11 and 25 <= day <= 29 and dt.weekday() == 0:
        return 1.8

    # Christmas Eve
    if month == 12 and day == 24:
        return 1.5

    # Christmas Day
    if month == 12 and day == 25:
        return 0.1

    # New Year's Eve
    if month == 12 and day == 31:
        return 1.2

    return 1.0


def _add_promo_price_features(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add promotion and price features.

    Args:
        df: DataFrame with sales data.
        seed: Random seed.

    Returns:
        DataFrame with promo and price features added.
    """
    np.random.seed(seed)

    # Generate base prices by item
    item_prices = {}
    for item in df["item_id"].unique():
        item_prices[item] = np.random.uniform(5, 50)

    # Add price and promo columns
    df["price"] = df["item_id"].map(item_prices)
    df["on_promo"] = 0

    # Add random promotions
    promo_probability = 0.1  # 10% chance of promotion on any given day

    for idx, row in df.iterrows():
        if np.random.random() < promo_probability:
            # Promotion effect
            promo_discount = np.random.uniform(0.1, 0.3)  # 10-30% discount
            df.loc[idx, "price"] *= (1 - promo_discount)
            df.loc[idx, "on_promo"] = 1

    # Add price elasticity effect on sales
    for idx, row in df.iterrows():
        if row["on_promo"]:
            # Increase sales during promotions
            elasticity = np.random.uniform(1.2, 2.0)  # 20-100% sales increase
            df.loc[idx, "sales"] *= elasticity

    return df


def _add_weather_features(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add weather features.

    Args:
        df: DataFrame with sales data.
        seed: Random seed.

    Returns:
        DataFrame with weather features added.
    """
    np.random.seed(seed)

    # Generate seasonal weather patterns
    df["temperature"] = None
    df["precipitation"] = None

    for idx, row in df.iterrows():
        dt = pd.to_datetime(row["date"])
        day_of_year = dt.timetuple().tm_yday

        # Temperature with seasonal pattern
        base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        temp_noise = np.random.normal(0, 5)
        df.loc[idx, "temperature"] = base_temp + temp_noise

        # Precipitation (higher in winter/spring)
        precip_base = 2 + 1.5 * np.sin(2 * np.pi * (day_of_year - 30) / 365.25)
        precip_noise = np.random.exponential(1)
        df.loc[idx, "precipitation"] = max(0, precip_base + precip_noise)

        # Weather effects on sales
        temp_effect = 1.0
        if df.loc[idx, "temperature"] < 0:  # Very cold
            temp_effect = 1.2
        elif df.loc[idx, "temperature"] > 30:  # Very hot
            temp_effect = 1.1

        precip_effect = 1.0
        if df.loc[idx, "precipitation"] > 10:  # Heavy rain
            precip_effect = 0.8

        df.loc[idx, "sales"] *= temp_effect * precip_effect

    return df


def _add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add holiday name features.

    Args:
        df: DataFrame with sales data.

    Returns:
        DataFrame with holiday features added.
    """
    df["holiday_name"] = None

    for idx, row in df.iterrows():
        dt = pd.to_datetime(row["date"])
        month = dt.month
        day = dt.day

        # Add holiday names
        if month == 1 and day == 1:
            df.loc[idx, "holiday_name"] = "New Year's Day"
        elif month == 2 and day == 14:
            df.loc[idx, "holiday_name"] = "Valentine's Day"
        elif month == 7 and day == 4:
            df.loc[idx, "holiday_name"] = "Independence Day"
        elif month == 10 and day == 31:
            df.loc[idx, "holiday_name"] = "Halloween"
        elif month == 12 and day == 25:
            df.loc[idx, "holiday_name"] = "Christmas Day"

    return df


def save_synthetic_data(
    df: pd.DataFrame,
    file_path: str = "data/raw/synth_sales.csv",
    settings: Settings = None,
) -> None:
    """Save synthetic data to file.

    Args:
        df: DataFrame to save.
        file_path: Path to save the data.
        settings: Application settings.
    """
    if settings is None:
        settings = Settings()

    # Ensure directory exists
    from pathlib import Path
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(file_path, index=False)

    logger.info("Saved synthetic data", file_path=file_path, shape=df.shape)


if __name__ == "__main__":
    # Generate and save synthetic data
    from forecastit.config.settings import Settings
    from forecastit.utils.logging import setup_logging

    settings = Settings()
    setup_logging(settings)

    # Generate data
    df = generate_synthetic_data(settings=settings)

    # Save data
    save_synthetic_data(df, settings=settings)
