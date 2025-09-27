"""Calendar feature engineering."""


import numpy as np
import pandas as pd
import structlog

from forecastit.utils.holidays import HolidayCalendar

logger = structlog.get_logger(__name__)


class CalendarFeatures:
    """Generate calendar-based features."""

    def __init__(self, country: str = "US", state: str | None = None):
        """Initialize calendar features.

        Args:
            country: Country code for holidays.
            state: State code for US holidays (optional).
        """
        self.holiday_calendar = HolidayCalendar(country=country, state=state)
        self.logger = logger.bind(component="calendar_features")

    def add_calendar_features(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
    ) -> pd.DataFrame:
        """Add calendar features to DataFrame.

        Args:
            data: DataFrame with date column.
            date_column: Name of the date column.

        Returns:
            DataFrame with calendar features added.
        """
        self.logger.info("Adding calendar features", shape=data.shape, date_column=date_column)

        # Make a copy to avoid modifying original data
        result = data.copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
            result[date_column] = pd.to_datetime(result[date_column])

        # Basic calendar features
        result = self._add_basic_calendar_features(result, date_column)

        # Holiday features
        result = self._add_holiday_features(result, date_column)

        # Seasonality features
        result = self._add_seasonality_features(result, date_column)

        self.logger.info(
            "Calendar features added",
            original_shape=data.shape,
            new_shape=result.shape,
            new_columns=set(result.columns) - set(data.columns),
        )

        return result

    def _add_basic_calendar_features(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add basic calendar features.

        Args:
            data: DataFrame with date column.
            date_column: Name of the date column.

        Returns:
            DataFrame with basic calendar features added.
        """
        dt = data[date_column].dt

        # Day features
        data["day_of_week"] = dt.dayofweek  # 0=Monday, 6=Sunday
        data["day_of_month"] = dt.day
        data["day_of_year"] = dt.dayofyear

        # Week features
        data["week_of_year"] = dt.isocalendar().week
        data["week_of_month"] = (dt.day - 1) // 7 + 1

        # Month features
        data["month"] = dt.month
        data["quarter"] = dt.quarter
        data["year"] = dt.year

        # Month boundaries
        data["is_month_start"] = dt.is_month_start
        data["is_month_end"] = dt.is_month_end
        data["is_quarter_start"] = dt.is_quarter_start
        data["is_quarter_end"] = dt.is_quarter_end
        data["is_year_start"] = dt.is_year_start
        data["is_year_end"] = dt.is_year_end

        # Weekend flag
        data["is_weekend"] = dt.dayofweek.isin([5, 6])  # Saturday, Sunday

        # Working day (Monday to Friday)
        data["is_working_day"] = dt.dayofweek.isin([0, 1, 2, 3, 4])

        return data

    def _add_holiday_features(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add holiday features.

        Args:
            data: DataFrame with date column.
            date_column: Name of the date column.

        Returns:
            DataFrame with holiday features added.
        """
        # Add holiday features using the holiday calendar
        data = self.holiday_calendar.add_holiday_features(
            data,
            date_column=date_column,
            prefix="holiday_",
        )

        # Add specific holiday flags
        data = self._add_specific_holiday_flags(data, date_column)

        return data

    def _add_specific_holiday_flags(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add flags for specific holidays.

        Args:
            data: DataFrame with date column.
            date_column: Name of the date column.

        Returns:
            DataFrame with specific holiday flags added.
        """
        dt = data[date_column].dt

        # New Year's Day
        data["is_new_year"] = (dt.month == 1) & (dt.day == 1)

        # Valentine's Day
        data["is_valentine"] = (dt.month == 2) & (dt.day == 14)

        # Easter (approximate - second Sunday in April)
        data["is_easter_period"] = (dt.month == 4) & (dt.day >= 10) & (dt.day <= 20)

        # Mother's Day (second Sunday in May)
        data["is_mothers_day_period"] = (dt.month == 5) & (dt.day >= 8) & (dt.day <= 14)

        # Father's Day (third Sunday in June)
        data["is_fathers_day_period"] = (dt.month == 6) & (dt.day >= 15) & (dt.day <= 21)

        # Independence Day
        data["is_independence_day"] = (dt.month == 7) & (dt.day == 4)

        # Halloween
        data["is_halloween"] = (dt.month == 10) & (dt.day == 31)

        # Thanksgiving (fourth Thursday in November)
        data["is_thanksgiving_period"] = (dt.month == 11) & (dt.day >= 22) & (dt.day <= 28)

        # Black Friday (day after Thanksgiving)
        data["is_black_friday_period"] = (dt.month == 11) & (dt.day >= 23) & (dt.day <= 29)

        # Christmas Eve and Day
        data["is_christmas_eve"] = (dt.month == 12) & (dt.day == 24)
        data["is_christmas"] = (dt.month == 12) & (dt.day == 25)

        # New Year's Eve
        data["is_new_year_eve"] = (dt.month == 12) & (dt.day == 31)

        # Holiday season (December)
        data["is_holiday_season"] = dt.month == 12

        return data

    def _add_seasonality_features(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add seasonality features.

        Args:
            data: DataFrame with date column.
            date_column: Name of the date column.

        Returns:
            DataFrame with seasonality features added.
        """
        dt = data[date_column].dt

        # Seasonal flags
        data["is_spring"] = dt.month.isin([3, 4, 5])
        data["is_summer"] = dt.month.isin([6, 7, 8])
        data["is_fall"] = dt.month.isin([9, 10, 11])
        data["is_winter"] = dt.month.isin([12, 1, 2])

        # Cyclical encoding for seasonality
        # Annual seasonality
        data["annual_sin"] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
        data["annual_cos"] = np.cos(2 * np.pi * dt.dayofyear / 365.25)

        # Monthly seasonality
        data["monthly_sin"] = np.sin(2 * np.pi * dt.month / 12)
        data["monthly_cos"] = np.cos(2 * np.pi * dt.month / 12)

        # Weekly seasonality
        data["weekly_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        data["weekly_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

        # Daily seasonality (if hour information is available)
        if hasattr(dt, "hour"):
            data["daily_sin"] = np.sin(2 * np.pi * dt.hour / 24)
            data["daily_cos"] = np.cos(2 * np.pi * dt.hour / 24)

        return data

    def get_calendar_feature_names(self) -> list[str]:
        """Get list of calendar feature names.

        Returns:
            List of calendar feature names.
        """
        return [
            # Basic calendar features
            "day_of_week", "day_of_month", "day_of_year",
            "week_of_year", "week_of_month",
            "month", "quarter", "year",
            "is_month_start", "is_month_end",
            "is_quarter_start", "is_quarter_end",
            "is_year_start", "is_year_end",
            "is_weekend", "is_working_day",

            # Holiday features
            "holiday_is_holiday", "holiday_holiday_name",
            "holiday_days_to_holiday", "holiday_days_from_holiday",
            "is_new_year", "is_valentine", "is_easter_period",
            "is_mothers_day_period", "is_fathers_day_period",
            "is_independence_day", "is_halloween",
            "is_thanksgiving_period", "is_black_friday_period",
            "is_christmas_eve", "is_christmas", "is_new_year_eve",
            "is_holiday_season",

            # Seasonality features
            "is_spring", "is_summer", "is_fall", "is_winter",
            "annual_sin", "annual_cos",
            "monthly_sin", "monthly_cos",
            "weekly_sin", "weekly_cos",
        ]
