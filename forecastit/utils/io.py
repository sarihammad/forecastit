"""I/O utilities for data loading and saving."""

from pathlib import Path
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def read_data(
    file_path: str | Path,
    file_type: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read data from various file formats.

    Args:
        file_path: Path to the data file.
        file_type: File type override (csv, parquet, json, excel).
        **kwargs: Additional arguments for pandas readers.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If file type is not supported.
        FileNotFoundError: If file doesn't exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip(".")

    logger.info("Reading data file", file_path=str(file_path), file_type=file_type)

    try:
        if file_type == "csv":
            return pd.read_csv(file_path, **kwargs)
        elif file_type == "parquet":
            return pd.read_parquet(file_path, **kwargs)
        elif file_type == "json":
            return pd.read_json(file_path, **kwargs)
        elif file_type in ["xlsx", "xls"]:
            return pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error("Failed to read data file", file_path=str(file_path), error=str(e))
        raise


def write_data(
    data: pd.DataFrame,
    file_path: str | Path,
    file_type: str | None = None,
    **kwargs: Any,
) -> None:
    """Write DataFrame to various file formats.

    Args:
        data: DataFrame to save.
        file_path: Path to save the data.
        file_type: File type override (csv, parquet, json, excel).
        **kwargs: Additional arguments for pandas writers.
    """
    file_path = Path(file_path)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine file type
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip(".")

    logger.info("Writing data file", file_path=str(file_path), file_type=file_type)

    try:
        if file_type == "csv":
            data.to_csv(file_path, index=False, **kwargs)
        elif file_type == "parquet":
            data.to_parquet(file_path, index=False, **kwargs)
        elif file_type == "json":
            data.to_json(file_path, orient="records", **kwargs)
        elif file_type in ["xlsx", "xls"]:
            data.to_excel(file_path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error("Failed to write data file", file_path=str(file_path), error=str(e))
        raise


def get_data_info(data: pd.DataFrame) -> dict[str, Any]:
    """Get information about a DataFrame.

    Args:
        data: DataFrame to analyze.

    Returns:
        Dictionary with data information.
    """
    return {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "memory_usage": data.memory_usage(deep=True).sum(),
        "missing_values": data.isnull().sum().to_dict(),
        "duplicate_rows": data.duplicated().sum(),
    }


def validate_data_schema(
    data: pd.DataFrame,
    required_columns: list[str],
    date_columns: list[str] | None = None,
) -> None:
    """Validate DataFrame schema.

    Args:
        data: DataFrame to validate.
        required_columns: List of required column names.
        date_columns: List of columns that should be datetime.

    Raises:
        ValueError: If schema validation fails.
    """
    # Check required columns
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check date columns
    if date_columns:
        for col in date_columns:
            if col in data.columns and not pd.api.types.is_datetime64_any_dtype(data[col]):
                try:
                    pd.to_datetime(data[col])
                except Exception as e:
                    raise ValueError(f"Column {col} is not a valid datetime: {e}")

    logger.info("Data schema validation passed", columns=list(data.columns))
