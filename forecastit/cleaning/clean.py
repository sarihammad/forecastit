"""Data cleaning and preprocessing utilities."""


import pandas as pd
import structlog

from forecastit.config.settings import Settings

logger = structlog.get_logger(__name__)


class DataCleaner:
    """Data cleaning and preprocessing utilities."""

    def __init__(self, settings: Settings):
        """Initialize data cleaner.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.logger = logger.bind(component="data_cleaner")

    def clean_data(
        self,
        data: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_outliers: bool = True,
        reindex_series: bool = True,
        impute_missing: bool = False,
    ) -> pd.DataFrame:
        """Clean and preprocess raw data.

        Args:
            data: Raw DataFrame to clean.
            remove_duplicates: Whether to remove duplicate records.
            handle_outliers: Whether to handle outliers.
            reindex_series: Whether to reindex time series for continuous dates.
            impute_missing: Whether to impute missing values.

        Returns:
            Cleaned DataFrame.
        """
        self.logger.info("Starting data cleaning", shape=data.shape)

        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()

        # Convert date column to datetime
        cleaned_data["date"] = pd.to_datetime(cleaned_data["date"])

        # Remove duplicates
        if remove_duplicates:
            cleaned_data = self._remove_duplicates(cleaned_data)

        # Handle outliers
        if handle_outliers:
            cleaned_data = self._handle_outliers(cleaned_data)

        # Reindex time series
        if reindex_series:
            cleaned_data = self._reindex_time_series(cleaned_data)

        # Impute missing values
        if impute_missing:
            cleaned_data = self._impute_missing_values(cleaned_data)

        # Add cleaning flags
        cleaned_data = self._add_cleaning_flags(cleaned_data, data)

        # Validate cleaned data
        self._validate_cleaned_data(cleaned_data)

        self.logger.info(
            "Data cleaning completed",
            original_shape=data.shape,
            cleaned_shape=cleaned_data.shape,
        )

        return cleaned_data

    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records.

        Args:
            data: DataFrame to clean.

        Returns:
            DataFrame with duplicates removed.
        """
        original_count = len(data)

        # Remove exact duplicates
        data = data.drop_duplicates()

        # Remove duplicates by key columns (date, store_id, item_id)
        if all(col in data.columns for col in ["date", "store_id", "item_id"]):
            data = data.drop_duplicates(subset=["date", "store_id", "item_id"], keep="first")

        removed_count = original_count - len(data)

        self.logger.info(
            "Removed duplicates",
            original_count=original_count,
            removed_count=removed_count,
            remaining_count=len(data),
        )

        return data

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numerical columns.

        Args:
            data: DataFrame to clean.

        Returns:
            DataFrame with outliers handled.
        """
        numerical_columns = ["sales", "price", "temperature", "precipitation"]
        outlier_counts = {}

        for col in numerical_columns:
            if col in data.columns:
                len(data)

                # IQR method for outlier detection
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers instead of removing them
                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    # Cap outliers to bounds
                    data.loc[data[col] < lower_bound, col] = lower_bound
                    data.loc[data[col] > upper_bound, col] = upper_bound

                    outlier_counts[col] = outlier_count

                    self.logger.info(
                        "Capped outliers",
                        column=col,
                        outlier_count=outlier_count,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )

        return data

    def _reindex_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reindex time series to have continuous daily data.

        Args:
            data: DataFrame to reindex.

        Returns:
            DataFrame with reindexed time series.
        """
        if not all(col in data.columns for col in ["date", "store_id", "item_id"]):
            self.logger.warning("Cannot reindex: missing required columns")
            return data

        # Get date range
        min_date = data["date"].min()
        max_date = data["date"].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        reindexed_data = []

        # Reindex each store-item combination
        for store in data["store_id"].unique():
            for item in data["item_id"].unique():
                subset = data[(data["store_id"] == store) & (data["item_id"] == item)]

                if len(subset) == 0:
                    continue

                # Create complete date index
                subset = subset.set_index("date")
                subset_reindexed = subset.reindex(date_range)

                # Reset index to get date column back
                subset_reindexed = subset_reindexed.reset_index()
                subset_reindexed.rename(columns={"index": "date"}, inplace=True)

                # Fill store_id and item_id
                subset_reindexed["store_id"] = store
                subset_reindexed["item_id"] = item

                reindexed_data.append(subset_reindexed)

        if reindexed_data:
            result = pd.concat(reindexed_data, ignore_index=True)

            # Sort by store, item, date
            result = result.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

            self.logger.info(
                "Reindexed time series",
                original_rows=len(data),
                reindexed_rows=len(result),
                date_range=f"{min_date} to {max_date}",
            )

            return result
        else:
            return data

    def _impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the dataset.

        Args:
            data: DataFrame to impute.

        Returns:
            DataFrame with missing values imputed.
        """
        imputed_data = data.copy()

        # Impute numerical columns
        numerical_columns = ["sales", "price", "temperature", "precipitation"]

        for col in numerical_columns:
            if col in imputed_data.columns:
                missing_count = imputed_data[col].isnull().sum()

                if missing_count > 0:
                    if col == "sales":
                        # For sales, use forward fill then backward fill
                        imputed_data[col] = imputed_data.groupby(["store_id", "item_id"])[col].fillna(method="ffill").fillna(method="bfill")
                    else:
                        # For other numerical columns, use median
                        imputed_data[col] = imputed_data[col].fillna(imputed_data[col].median())

                    self.logger.info(
                        "Imputed missing values",
                        column=col,
                        missing_count=missing_count,
                        method="forward_fill" if col == "sales" else "median",
                    )

        # Impute categorical columns
        categorical_columns = ["on_promo", "holiday_name"]

        for col in categorical_columns:
            if col in imputed_data.columns:
                missing_count = imputed_data[col].isnull().sum()

                if missing_count > 0:
                    if col == "on_promo":
                        # Default to 0 (no promotion)
                        imputed_data[col] = imputed_data[col].fillna(0)
                    else:
                        # Default to None for holiday names
                        imputed_data[col] = imputed_data[col].fillna("")

                    self.logger.info(
                        "Imputed missing values",
                        column=col,
                        missing_count=missing_count,
                        method="default_value",
                    )

        return imputed_data

    def _add_cleaning_flags(self, cleaned_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Add flags indicating data cleaning operations.

        Args:
            cleaned_data: Cleaned DataFrame.
            original_data: Original DataFrame.

        Returns:
            DataFrame with cleaning flags added.
        """
        cleaned_data = cleaned_data.copy()

        # Initialize flags
        cleaned_data["is_outlier"] = False
        cleaned_data["is_missing"] = False

        # Check for missing values in key columns
        if "sales" in cleaned_data.columns:
            cleaned_data["is_missing"] = cleaned_data["sales"].isnull()

        # Check for outliers in sales (using original data if available)
        if "sales" in cleaned_data.columns:
            # Use IQR method to identify outliers
            Q1 = cleaned_data["sales"].quantile(0.25)
            Q3 = cleaned_data["sales"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Mark outliers (but don't remove them)
            outlier_mask = (cleaned_data["sales"] < lower_bound) | (cleaned_data["sales"] > upper_bound)
            cleaned_data.loc[outlier_mask, "is_outlier"] = True

        return cleaned_data

    def _validate_cleaned_data(self, data: pd.DataFrame) -> None:
        """Validate cleaned data.

        Args:
            data: DataFrame to validate.

        Raises:
            ValueError: If validation fails.
        """
        # Check required columns
        required_columns = ["date", "store_id", "item_id"]
        missing_columns = set(required_columns) - set(data.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns after cleaning: {missing_columns}")

        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data["date"]):
            raise ValueError("Date column is not datetime type")

        # Check for negative sales
        if "sales" in data.columns:
            negative_sales = (data["sales"] < 0).sum()
            if negative_sales > 0:
                self.logger.warning(f"Found {negative_sales} negative sales values")

        # Check promotion flags
        if "on_promo" in data.columns:
            invalid_promo = ~data["on_promo"].isin([0, 1])
            if invalid_promo.sum() > 0:
                raise ValueError(f"Found {invalid_promo.sum()} invalid promotion flags")

        self.logger.info("Cleaned data validation passed")

    def get_cleaning_summary(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> dict:
        """Get summary of cleaning operations.

        Args:
            original_data: Original DataFrame.
            cleaned_data: Cleaned DataFrame.

        Returns:
            Dictionary with cleaning summary.
        """
        return {
            "original_shape": original_data.shape,
            "cleaned_shape": cleaned_data.shape,
            "rows_removed": original_data.shape[0] - cleaned_data.shape[0],
            "columns_modified": len(cleaned_data.columns),
            "missing_values": {
                "original": original_data.isnull().sum().to_dict(),
                "cleaned": cleaned_data.isnull().sum().to_dict(),
            },
            "outliers_flagged": cleaned_data.get("is_outlier", pd.Series()).sum() if "is_outlier" in cleaned_data.columns else 0,
            "missing_flagged": cleaned_data.get("is_missing", pd.Series()).sum() if "is_missing" in cleaned_data.columns else 0,
        }
