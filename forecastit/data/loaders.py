"""Data loading utilities for various sources."""

from pathlib import Path

import pandas as pd
import structlog

from forecastit.config.settings import Settings
from forecastit.utils.io import read_data, validate_data_schema

logger = structlog.get_logger(__name__)


class DataLoader:
    """Data loader for various sources."""

    def __init__(self, settings: Settings):
        """Initialize data loader.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.logger = logger.bind(component="data_loader")

    def load_local(
        self,
        file_path: str | Path,
        file_type: str | None = None,
        validate_schema: bool = True,
    ) -> pd.DataFrame:
        """Load data from local file.

        Args:
            file_path: Path to the data file.
            file_type: File type override.
            validate_schema: Whether to validate data schema.

        Returns:
            Loaded DataFrame.
        """
        self.logger.info("Loading local data", file_path=str(file_path))

        try:
            data = read_data(file_path, file_type=file_type)

            if validate_schema:
                self._validate_raw_schema(data)

            self.logger.info(
                "Successfully loaded local data",
                file_path=str(file_path),
                shape=data.shape,
            )

            return data

        except Exception as e:
            self.logger.error("Failed to load local data", file_path=str(file_path), error=str(e))
            raise

    def load_kaggle(
        self,
        dataset: str,
        filename: str,
        validate_schema: bool = True,
    ) -> pd.DataFrame:
        """Load data from Kaggle dataset.

        Args:
            dataset: Kaggle dataset identifier (username/dataset).
            filename: Name of the file to download.
            validate_schema: Whether to validate data schema.

        Returns:
            Loaded DataFrame.
        """
        self.logger.info("Loading Kaggle data", dataset=dataset, filename=filename)

        # Check for Kaggle credentials
        if not self.settings.kaggle_username or not self.settings.kaggle_key:
            self.logger.warning(
                "Kaggle credentials not found. Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
            )
            raise ValueError("Kaggle credentials not configured")

        try:
            # Set up Kaggle credentials
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)

            kaggle_json = {
                "username": self.settings.kaggle_username,
                "key": self.settings.kaggle_key,
            }

            import json
            with open(kaggle_dir / "kaggle.json", "w") as f:
                json.dump(kaggle_json, f)

            # Download dataset
            import kaggle
            kaggle.api.dataset_download_file(
                dataset=dataset,
                file_name=filename,
                path=str(self.settings.raw_data_dir),
                unzip=True,
            )

            # Load downloaded file
            file_path = self.settings.raw_data_dir / filename
            data = read_data(file_path)

            if validate_schema:
                self._validate_raw_schema(data)

            self.logger.info(
                "Successfully loaded Kaggle data",
                dataset=dataset,
                filename=filename,
                shape=data.shape,
            )

            return data

        except Exception as e:
            self.logger.error(
                "Failed to load Kaggle data",
                dataset=dataset,
                filename=filename,
                error=str(e),
            )
            raise

    def load_synthetic(
        self,
        validate_schema: bool = True,
    ) -> pd.DataFrame:
        """Load synthetic data.

        Args:
            validate_schema: Whether to validate data schema.

        Returns:
            Loaded DataFrame.
        """
        self.logger.info("Loading synthetic data")

        try:
            from forecastit.data.make_synthetic import generate_synthetic_data

            # Check if synthetic data already exists
            synth_file = self.settings.raw_data_dir / "synth_sales.csv"

            if synth_file.exists():
                self.logger.info("Loading existing synthetic data", file_path=str(synth_file))
                data = read_data(synth_file)
            else:
                self.logger.info("Generating new synthetic data")
                data = generate_synthetic_data(settings=self.settings)

                # Save generated data
                from forecastit.data.make_synthetic import save_synthetic_data
                save_synthetic_data(data, str(synth_file), self.settings)

            if validate_schema:
                self._validate_raw_schema(data)

            self.logger.info("Successfully loaded synthetic data", shape=data.shape)

            return data

        except Exception as e:
            self.logger.error("Failed to load synthetic data", error=str(e))
            raise

    def load_from_directory(
        self,
        directory: str | Path,
        file_pattern: str = "*.csv",
        validate_schema: bool = True,
    ) -> pd.DataFrame:
        """Load and combine multiple files from directory.

        Args:
            directory: Directory containing data files.
            file_pattern: File pattern to match.
            validate_schema: Whether to validate data schema.

        Returns:
            Combined DataFrame.
        """
        directory = Path(directory)

        self.logger.info("Loading data from directory", directory=str(directory), pattern=file_pattern)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find matching files
        files = list(directory.glob(file_pattern))

        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

        self.logger.info("Found files", count=len(files), files=[str(f) for f in files])

        # Load and combine files
        dataframes = []

        for file_path in files:
            try:
                df = read_data(file_path)
                dataframes.append(df)

                self.logger.info("Loaded file", file_path=str(file_path), shape=df.shape)

            except Exception as e:
                self.logger.warning("Failed to load file", file_path=str(file_path), error=str(e))
                continue

        if not dataframes:
            raise ValueError("No files could be loaded successfully")

        # Combine dataframes
        combined_data = pd.concat(dataframes, ignore_index=True)

        if validate_schema:
            self._validate_raw_schema(combined_data)

        self.logger.info(
            "Successfully combined data from directory",
            directory=str(directory),
            total_shape=combined_data.shape,
            files_loaded=len(dataframes),
        )

        return combined_data

    def _validate_raw_schema(self, data: pd.DataFrame) -> None:
        """Validate raw data schema.

        Args:
            data: DataFrame to validate.

        Raises:
            ValueError: If schema validation fails.
        """
        required_columns = ["date", "store_id", "item_id", "sales", "on_promo"]
        validate_data_schema(
            data,
            required_columns=required_columns,
            date_columns=["date"],
        )

        # Additional validation
        if data["sales"].min() < 0:
            raise ValueError("Sales values must be non-negative")

        if not data["on_promo"].isin([0, 1]).all():
            raise ValueError("on_promo must contain only 0 and 1 values")

        self.logger.info("Raw data schema validation passed")

    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """Get summary statistics for loaded data.

        Args:
            data: DataFrame to summarize.

        Returns:
            Dictionary with summary statistics.
        """
        summary = {
            "shape": data.shape,
            "date_range": {
                "start": data["date"].min(),
                "end": data["date"].max(),
                "days": (data["date"].max() - data["date"].min()).days,
            },
            "stores": {
                "count": data["store_id"].nunique(),
                "ids": list(data["store_id"].unique()),
            },
            "items": {
                "count": data["item_id"].nunique(),
                "ids": list(data["item_id"].unique())[:10],  # First 10 items
            },
            "sales": {
                "total": data["sales"].sum(),
                "mean": data["sales"].mean(),
                "median": data["sales"].median(),
                "std": data["sales"].std(),
                "min": data["sales"].min(),
                "max": data["sales"].max(),
            },
            "promotions": {
                "total_promo_days": data["on_promo"].sum(),
                "promo_percentage": (data["on_promo"].mean() * 100),
            },
            "missing_values": data.isnull().sum().to_dict(),
        }

        return summary


def load_data(
    source: str = "synthetic",
    **kwargs,
) -> pd.DataFrame:
    """Convenience function to load data from various sources.

    Args:
        source: Data source ("local", "kaggle", "synthetic").
        **kwargs: Additional arguments for the loader.

    Returns:
        Loaded DataFrame.
    """
    settings = Settings()
    loader = DataLoader(settings)

    if source == "synthetic":
        return loader.load_synthetic(**kwargs)
    elif source == "local":
        return loader.load_local(**kwargs)
    elif source == "kaggle":
        return loader.load_kaggle(**kwargs)
    else:
        raise ValueError(f"Unknown data source: {source}")
