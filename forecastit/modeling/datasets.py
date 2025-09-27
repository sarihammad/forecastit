"""Time series dataset handling and splitting utilities."""


import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import TimeSeriesSplit

logger = structlog.get_logger(__name__)


class TimeSeriesDataset:
    """Time series dataset with proper splitting for forecasting."""

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str = "sales",
        group_columns: list[str] | None = None,
        date_column: str = "date",
    ):
        """Initialize time series dataset.

        Args:
            data: DataFrame with time series data.
            target_column: Name of the target column.
            group_columns: Columns to group by (e.g., ['store_id', 'item_id']).
            date_column: Name of the date column.
        """
        self.data = data.copy()
        self.target_column = target_column
        self.group_columns = group_columns or ["store_id", "item_id"]
        self.date_column = date_column
        self.logger = logger.bind(component="timeseries_dataset")

        # Sort data by group and date
        sort_columns = [*self.group_columns, self.date_column]
        self.data = self.data.sort_values(sort_columns).reset_index(drop=True)

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])

        self.logger.info(
            "Initialized time series dataset",
            shape=self.data.shape,
            target_column=target_column,
            group_columns=self.group_columns,
            date_range=f"{self.data[self.date_column].min()} to {self.data[self.date_column].max()}",
        )

    def get_time_series_split(
        self,
        n_splits: int = 5,
        test_size: float | None = None,
        gap: int = 0,
    ) -> TimeSeriesSplit:
        """Get time series split for cross-validation.

        Args:
            n_splits: Number of splits.
            test_size: Size of test set (fraction of total data).
            gap: Gap between train and test sets.

        Returns:
            TimeSeriesSplit object.
        """
        return TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
        )

    def split_by_date(
        self,
        train_end_date: str,
        val_start_date: str | None = None,
        test_start_date: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Split dataset by date.

        Args:
            train_end_date: End date for training set.
            val_start_date: Start date for validation set.
            test_start_date: Start date for test set.

        Returns:
            Tuple of (train, val, test) DataFrames.
        """
        train_end_date = pd.to_datetime(train_end_date)
        val_start_date = pd.to_datetime(val_start_date) if val_start_date else None
        test_start_date = pd.to_datetime(test_start_date) if test_start_date else None

        # Training set
        train_data = self.data[self.data[self.date_column] <= train_end_date].copy()

        # Validation set
        val_data = None
        if val_start_date:
            val_end_date = test_start_date if test_start_date else self.data[self.date_column].max()
            val_data = self.data[
                (self.data[self.date_column] >= val_start_date) &
                (self.data[self.date_column] < val_end_date)
            ].copy()

        # Test set
        test_data = None
        if test_start_date:
            test_data = self.data[self.data[self.date_column] >= test_start_date].copy()

        self.logger.info(
            "Split dataset by date",
            train_shape=train_data.shape,
            val_shape=val_data.shape if val_data is not None else None,
            test_shape=test_data.shape if test_data is not None else None,
        )

        return train_data, val_data, test_data

    def split_by_series(
        self,
        train_series: list[tuple[str, ...]],
        val_series: list[tuple[str, ...]] | None = None,
        test_series: list[tuple[str, ...]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Split dataset by series (store-item combinations).

        Args:
            train_series: List of series tuples for training.
            val_series: List of series tuples for validation.
            test_series: List of series tuples for testing.

        Returns:
            Tuple of (train, val, test) DataFrames.
        """
        # Training set
        train_mask = self.data.apply(
            lambda row: tuple(row[col] for col in self.group_columns) in train_series,
            axis=1
        )
        train_data = self.data[train_mask].copy()

        # Validation set
        val_data = None
        if val_series:
            val_mask = self.data.apply(
                lambda row: tuple(row[col] for col in self.group_columns) in val_series,
                axis=1
            )
            val_data = self.data[val_mask].copy()

        # Test set
        test_data = None
        if test_series:
            test_mask = self.data.apply(
                lambda row: tuple(row[col] for col in self.group_columns) in test_series,
                axis=1
            )
            test_data = self.data[test_mask].copy()

        self.logger.info(
            "Split dataset by series",
            train_shape=train_data.shape,
            val_shape=val_data.shape if val_data is not None else None,
            test_shape=test_data.shape if test_data is not None else None,
        )

        return train_data, val_data, test_data

    def get_series_info(self) -> pd.DataFrame:
        """Get information about each time series.

        Returns:
            DataFrame with series information.
        """
        series_info = []

        for group, group_data in self.data.groupby(self.group_columns):
            series_info.append({
                "series_key": "_".join(str(g) for g in group),
                "store_id": group[0] if len(group) > 0 else None,
                "item_id": group[1] if len(group) > 1 else None,
                "start_date": group_data[self.date_column].min(),
                "end_date": group_data[self.date_column].max(),
                "length": len(group_data),
                "missing_values": group_data[self.target_column].isnull().sum(),
                "mean_sales": group_data[self.target_column].mean(),
                "std_sales": group_data[self.target_column].std(),
                "min_sales": group_data[self.target_column].min(),
                "max_sales": group_data[self.target_column].max(),
            })

        return pd.DataFrame(series_info)

    def get_feature_columns(self, exclude_columns: list[str] | None = None) -> list[str]:
        """Get list of feature columns.

        Args:
            exclude_columns: Columns to exclude from features.

        Returns:
            List of feature column names.
        """
        if exclude_columns is None:
            exclude_columns = [self.target_column, self.date_column, *self.group_columns]

        feature_columns = [col for col in self.data.columns if col not in exclude_columns]

        # Filter out non-numeric columns (except categorical)
        numeric_columns = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                numeric_columns.append(col)
            elif self.data[col].dtype == "object" or self.data[col].dtype.name == "category":
                # Keep categorical columns
                numeric_columns.append(col)

        return numeric_columns

    def prepare_for_modeling(
        self,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for modeling.

        Args:
            feature_columns: List of feature columns to use.
            target_column: Name of target column.

        Returns:
            Tuple of (X, y) arrays.
        """
        if feature_columns is None:
            feature_columns = self.get_feature_columns()

        if target_column is None:
            target_column = self.target_column

        # Select features and target
        X = self.data[feature_columns].values
        y = self.data[target_column].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        self.logger.info(
            "Prepared data for modeling",
            X_shape=X.shape,
            y_shape=y.shape,
            feature_count=len(feature_columns),
        )

        return X, y

    def get_series_groups(self) -> list[tuple[str, ...]]:
        """Get list of unique series groups.

        Returns:
            List of series group tuples.
        """
        return list(self.data.groupby(self.group_columns).groups.keys())

    def filter_by_series(
        self,
        series_list: list[tuple[str, ...]],
    ) -> pd.DataFrame:
        """Filter dataset by specific series.

        Args:
            series_list: List of series to keep.

        Returns:
            Filtered DataFrame.
        """
        mask = self.data.apply(
            lambda row: tuple(row[col] for col in self.group_columns) in series_list,
            axis=1
        )

        filtered_data = self.data[mask].copy()

        self.logger.info(
            "Filtered dataset by series",
            original_shape=self.data.shape,
            filtered_shape=filtered_data.shape,
            series_count=len(series_list),
        )

        return filtered_data

    def get_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Get date range of the dataset.

        Returns:
            Tuple of (start_date, end_date).
        """
        return self.data[self.date_column].min(), self.data[self.date_column].max()

    def get_series_count(self) -> int:
        """Get number of unique series.

        Returns:
            Number of unique series.
        """
        return self.data.groupby(self.group_columns).ngroups

    def validate_data(self) -> bool:
        """Validate dataset integrity.

        Returns:
            True if data is valid.
        """
        # Check for required columns
        required_columns = [self.date_column, self.target_column, *self.group_columns]
        missing_columns = set(required_columns) - set(self.data.columns)

        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check for duplicate dates within series
        duplicates = self.data.groupby([*self.group_columns, self.date_column]).size()
        if (duplicates > 1).any():
            self.logger.error("Found duplicate dates within series")
            return False

        # Check for missing values in key columns
        key_columns = [self.date_column, *self.group_columns]
        for col in key_columns:
            if self.data[col].isnull().any():
                self.logger.error(f"Found null values in key column: {col}")
                return False

        self.logger.info("Dataset validation passed")
        return True
