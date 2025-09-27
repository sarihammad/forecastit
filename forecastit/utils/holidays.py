"""Holiday calendar utilities."""


import holidays
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class HolidayCalendar:
    """Holiday calendar for different countries/regions."""

    def __init__(self, country: str = "US", state: str | None = None):
        """Initialize holiday calendar.

        Args:
            country: Country code for holidays.
            state: State code for US holidays (optional).
        """
        self.country = country
        self.state = state
        self._holidays = holidays.country_holidays(country, state=state)

        logger.info(
            "Initialized holiday calendar",
            country=country,
            state=state,
            num_holidays=len(self._holidays),
        )

    def get_holidays(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> dict[str, str]:
        """Get holidays in date range.

        Args:
            start_date: Start date for holiday lookup.
            end_date: End date for holiday lookup.

        Returns:
            Dictionary mapping dates to holiday names.
        """
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()

        holidays_dict = {}
        for date, name in self._holidays.items():
            if start_date <= date <= end_date:
                holidays_dict[date.strftime("%Y-%m-%d")] = name

        return holidays_dict

    def is_holiday(self, date: str | pd.Timestamp) -> bool:
        """Check if a date is a holiday.

        Args:
            date: Date to check.

        Returns:
            True if date is a holiday.
        """
        date_obj = pd.to_datetime(date).date()
        return date_obj in self._holidays

    def get_holiday_name(self, date: str | pd.Timestamp) -> str | None:
        """Get holiday name for a date.

        Args:
            date: Date to check.

        Returns:
            Holiday name if date is a holiday, None otherwise.
        """
        date_obj = pd.to_datetime(date).date()
        return self._holidays.get(date_obj)

    def add_holiday_features(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
        prefix: str = "holiday_",
    ) -> pd.DataFrame:
        """Add holiday features to DataFrame.

        Args:
            data: DataFrame with date column.
            date_column: Name of the date column.
            prefix: Prefix for holiday feature columns.

        Returns:
            DataFrame with holiday features added.
        """
        data = data.copy()

        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])

        # Add holiday indicators
        data[f"{prefix}is_holiday"] = data[date_column].dt.date.isin(self._holidays)
        data[f"{prefix}holiday_name"] = data[date_column].dt.date.map(
            lambda x: self._holidays.get(x) if x in self._holidays else None
        )

        # Add days to/from holidays
        data[f"{prefix}days_to_holiday"] = data[date_column].apply(
            self._days_to_next_holiday
        )
        data[f"{prefix}days_from_holiday"] = data[date_column].apply(
            self._days_from_last_holiday
        )

        logger.info(
            "Added holiday features",
            num_rows=len(data),
            prefix=prefix,
            holiday_count=data[f"{prefix}is_holiday"].sum(),
        )

        return data

    def _days_to_next_holiday(self, date: pd.Timestamp) -> int:
        """Calculate days to next holiday.

        Args:
            date: Date to check from.

        Returns:
            Days to next holiday.
        """
        date_obj = date.date()
        for holiday_date in sorted(self._holidays.keys()):
            if holiday_date > date_obj:
                return (holiday_date - date_obj).days
        return 365  # No holidays in next year

    def _days_from_last_holiday(self, date: pd.Timestamp) -> int:
        """Calculate days from last holiday.

        Args:
            date: Date to check from.

        Returns:
            Days from last holiday.
        """
        date_obj = date.date()
        for holiday_date in sorted(self._holidays.keys(), reverse=True):
            if holiday_date < date_obj:
                return (date_obj - holiday_date).days
        return 365  # No holidays in past year


def get_common_holidays() -> list[str]:
    """Get list of common holiday names across countries.

    Returns:
        List of common holiday names.
    """
    return [
        "New Year's Day",
        "Christmas Day",
        "Thanksgiving",
        "Black Friday",
        "Cyber Monday",
        "Valentine's Day",
        "Easter",
        "Mother's Day",
        "Father's Day",
        "Independence Day",
        "Labor Day",
        "Memorial Day",
        "Halloween",
    ]
