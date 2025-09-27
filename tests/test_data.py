"""Tests for data handling modules."""

from datetime import date

import pandas as pd

from forecastit.cleaning.clean import DataCleaner
from forecastit.config.settings import Settings
from forecastit.data.loaders import DataLoader
from forecastit.data.make_synthetic import generate_synthetic_data


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        settings = Settings()

        # Generate small dataset
        data = generate_synthetic_data(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            stores=["store_01", "store_02"],
            items=["item_001", "item_002"],
            seed=42,
            settings=settings,
        )

        # Check basic properties
        assert len(data) > 0
        assert "date" in data.columns
        assert "store_id" in data.columns
        assert "item_id" in data.columns
        assert "sales" in data.columns
        assert "on_promo" in data.columns
        assert "price" in data.columns

        # Check data types - date column can be object (Python date) or datetime64
        assert data["date"].dtype in ["object", "datetime64[ns]"]
        assert data["sales"].dtype == "float64"
        assert data["on_promo"].dtype == "int64"

        # Check data quality
        assert data["sales"].min() >= 0
        assert data["on_promo"].isin([0, 1]).all()
        assert data["price"].min() > 0

    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data generation is reproducible."""
        settings = Settings()

        # Generate data twice with same seed
        data1 = generate_synthetic_data(seed=42, settings=settings)
        data2 = generate_synthetic_data(seed=42, settings=settings)

        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)


class TestDataLoader:
    """Test data loading functionality."""

    def test_load_synthetic(self):
        """Test loading synthetic data."""
        settings = Settings()
        loader = DataLoader(settings)

        data = loader.load_synthetic()

        assert len(data) > 0
        assert "date" in data.columns
        assert "store_id" in data.columns
        assert "item_id" in data.columns

    def test_data_summary(self):
        """Test data summary generation."""
        settings = Settings()
        loader = DataLoader(settings)

        # Generate test data
        data = generate_synthetic_data(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
            stores=["store_01"],
            items=["item_001"],
            settings=settings,
        )

        summary = loader.get_data_summary(data)

        assert "shape" in summary
        assert "date_range" in summary
        assert "stores" in summary
        assert "items" in summary
        assert "sales" in summary


class TestDataCleaner:
    """Test data cleaning functionality."""

    def test_clean_data(self):
        """Test data cleaning."""
        settings = Settings()
        cleaner = DataCleaner(settings)

        # Generate test data
        data = generate_synthetic_data(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
            stores=["store_01"],
            items=["item_001"],
            settings=settings,
        )

        # Clean data
        cleaned_data = cleaner.clean_data(data)

        # Check that cleaning flags were added
        assert "is_outlier" in cleaned_data.columns
        assert "is_missing" in cleaned_data.columns

        # Check data quality
        assert cleaned_data["sales"].min() >= 0
        assert cleaned_data["on_promo"].isin([0, 1]).all()

    def test_remove_duplicates(self):
        """Test duplicate removal."""
        settings = Settings()
        cleaner = DataCleaner(settings)

        # Create data with duplicates
        data = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "store_id": ["store_01", "store_01", "store_01"],
            "item_id": ["item_001", "item_001", "item_001"],
            "sales": [100, 100, 150],
            "on_promo": [0, 0, 1],
        })
        data["date"] = pd.to_datetime(data["date"])

        # Clean data
        cleaned_data = cleaner._remove_duplicates(data)

        # Should have removed one duplicate
        assert len(cleaned_data) == 2
        assert cleaned_data["date"].nunique() == 2

    def test_handle_outliers(self):
        """Test outlier handling."""
        settings = Settings()
        cleaner = DataCleaner(settings)

        # Create data with outliers
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "store_id": ["store_01"] * 10,
            "item_id": ["item_001"] * 10,
            "sales": [100, 100, 100, 1000, 100, 100, 100, 100, 100, 100],  # 1000 is outlier
            "on_promo": [0] * 10,
        })

        # Clean data
        cleaned_data = cleaner._handle_outliers(data)

        # Outlier should be capped
        assert cleaned_data["sales"].max() < 1000
        assert cleaned_data["sales"].min() >= 0
