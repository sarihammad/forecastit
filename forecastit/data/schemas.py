"""Data schemas and validation models."""

from datetime import date, datetime
from typing import Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, validator


class RawDataSchema(BaseModel):
    """Schema for raw sales data."""

    date: str = Field(..., description="Date of the record")
    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item/SKU identifier")
    sales: float = Field(..., ge=0, description="Sales amount (non-negative)")
    on_promo: int = Field(..., ge=0, le=1, description="Promotion flag (0 or 1)")
    price: Optional[float] = Field(None, gt=0, description="Item price (positive)")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    precipitation: Optional[float] = Field(None, ge=0, description="Precipitation in mm")
    holiday_name: Optional[str] = Field(None, description="Holiday name if applicable")

    @validator("date", pre=True)
    def parse_date(cls, v):
        """Parse date from various formats."""
        if isinstance(v, str):
            try:
                return pd.to_datetime(v).date()
            except Exception:
                raise ValueError(f"Invalid date format: {v}")
        return v

    @validator("sales")
    def validate_sales(cls, v):
        """Validate sales is non-negative."""
        if v < 0:
            raise ValueError("Sales must be non-negative")
        return v

    @validator("price")
    def validate_price(cls, v):
        """Validate price is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return v


class CleanDataSchema(BaseModel):
    """Schema for cleaned sales data."""

    date: str = Field(..., description="Date of the record")
    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item/SKU identifier")
    sales: Optional[float] = Field(None, ge=0, description="Sales amount (can be NaN)")
    on_promo: int = Field(..., ge=0, le=1, description="Promotion flag (0 or 1)")
    price: Optional[float] = Field(None, gt=0, description="Item price")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    precipitation: Optional[float] = Field(None, ge=0, description="Precipitation in mm")
    holiday_name: Optional[str] = Field(None, description="Holiday name if applicable")
    is_outlier: bool = Field(default=False, description="Outlier flag")
    is_missing: bool = Field(default=False, description="Missing data flag")

    @validator("date", pre=True)
    def parse_date(cls, v):
        """Parse date from various formats."""
        if isinstance(v, str):
            try:
                return pd.to_datetime(v).date()
            except Exception:
                raise ValueError(f"Invalid date format: {v}")
        return v


class FeatureSchema(BaseModel):
    """Schema for engineered features."""

    # Basic identifiers
    date: str = Field(..., description="Date of the record")
    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item/SKU identifier")

    # Target variable
    sales: Optional[float] = Field(None, description="Sales amount")

    # Calendar features
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month")
    week_of_year: int = Field(..., ge=1, le=53, description="Week of year")
    month: int = Field(..., ge=1, le=12, description="Month")
    quarter: int = Field(..., ge=1, le=4, description="Quarter")
    year: int = Field(..., ge=2020, le=2030, description="Year")
    is_month_start: bool = Field(..., description="Is month start")
    is_month_end: bool = Field(..., description="Is month end")
    is_weekend: bool = Field(..., description="Is weekend")

    # Holiday features
    is_holiday: bool = Field(..., description="Is holiday")
    holiday_name: Optional[str] = Field(None, description="Holiday name")
    days_to_holiday: int = Field(..., description="Days to next holiday")
    days_from_holiday: int = Field(..., description="Days from last holiday")

    # Lag features
    sales_lag_1: Optional[float] = Field(None, description="Sales lag 1 day")
    sales_lag_7: Optional[float] = Field(None, description="Sales lag 7 days")
    sales_lag_14: Optional[float] = Field(None, description="Sales lag 14 days")
    sales_lag_28: Optional[float] = Field(None, description="Sales lag 28 days")

    # Rolling features
    sales_rolling_mean_7: Optional[float] = Field(None, description="7-day rolling mean")
    sales_rolling_mean_14: Optional[float] = Field(None, description="14-day rolling mean")
    sales_rolling_mean_28: Optional[float] = Field(None, description="28-day rolling mean")
    sales_rolling_std_7: Optional[float] = Field(None, description="7-day rolling std")
    sales_rolling_std_14: Optional[float] = Field(None, description="14-day rolling std")
    sales_rolling_std_28: Optional[float] = Field(None, description="28-day rolling std")
    sales_ema_14: Optional[float] = Field(None, description="14-day EMA")

    # Promo/Price features
    on_promo: int = Field(..., ge=0, le=1, description="Promotion flag")
    price: Optional[float] = Field(None, gt=0, description="Item price")
    price_pct_change: Optional[float] = Field(None, description="Price percentage change")
    promo_rolling_7: float = Field(..., ge=0, le=7, description="7-day promo count")

    # External features
    temperature: Optional[float] = Field(None, description="Temperature")
    precipitation: Optional[float] = Field(None, ge=0, description="Precipitation")
    temp_lag_1: Optional[float] = Field(None, description="Temperature lag 1")
    temp_lag_7: Optional[float] = Field(None, description="Temperature lag 7")
    precip_lag_1: Optional[float] = Field(None, description="Precipitation lag 1")
    precip_lag_7: Optional[float] = Field(None, description="Precipitation lag 7")

    # Seasonality features
    is_spring: bool = Field(..., description="Is spring season")
    is_summer: bool = Field(..., description="Is summer season")
    is_fall: bool = Field(..., description="Is fall season")
    is_winter: bool = Field(..., description="Is winter season")

    @validator("date", pre=True)
    def parse_date(cls, v):
        """Parse date from various formats."""
        if isinstance(v, str):
            try:
                return pd.to_datetime(v).date()
            except Exception:
                raise ValueError(f"Invalid date format: {v}")
        return v


class ForecastRequest(BaseModel):
    """Schema for forecast API requests."""

    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item identifier")
    start_date: Union[date, datetime, str] = Field(..., description="Forecast start date")
    end_date: Union[date, datetime, str] = Field(..., description="Forecast end date")
    promo_plan: Optional[list[int]] = Field(None, description="Promo flags for forecast period")
    price_plan: Optional[list[float]] = Field(None, description="Price plan for forecast period")

    @validator("start_date", "end_date", pre=True)
    def parse_date(cls, v):
        """Parse date from various formats."""
        if isinstance(v, str):
            try:
                return pd.to_datetime(v).date()
            except Exception:
                raise ValueError(f"Invalid date format: {v}")
        return v

    @validator("end_date")
    def validate_date_range(cls, v, values):
        """Validate end date is after start date."""
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("End date must be after start date")
        return v


class ForecastResponse(BaseModel):
    """Schema for forecast API responses."""

    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item identifier")
    forecast_date: Union[date, datetime, str] = Field(..., description="Forecast date")
    yhat: float = Field(..., description="Point forecast")
    yhat_lower: float = Field(..., description="Lower confidence bound")
    yhat_upper: float = Field(..., description="Upper confidence bound")
    model_name: str = Field(..., description="Model used for forecast")
    model_version: str = Field(..., description="Model version")
    created_at: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    @validator("forecast_date", pre=True)
    def parse_date(cls, v):
        """Parse date from various formats."""
        if isinstance(v, str):
            try:
                return pd.to_datetime(v).date()
            except Exception:
                raise ValueError(f"Invalid date format: {v}")
        return v


class InventoryRequest(BaseModel):
    """Schema for inventory optimization requests."""

    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item identifier")
    service_level: float = Field(default=0.95, ge=0.5, le=1.0, description="Service level")
    lead_time: int = Field(default=7, ge=1, description="Lead time in days")
    holding_cost: float = Field(default=0.1, gt=0, description="Holding cost per unit per day")
    stockout_cost: float = Field(default=10.0, gt=0, description="Stockout cost per unit")


class InventoryResponse(BaseModel):
    """Schema for inventory optimization responses."""

    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item identifier")
    reorder_point: float = Field(..., description="Recommended reorder point")
    safety_stock: float = Field(..., description="Recommended safety stock")
    expected_demand: float = Field(..., description="Expected demand during lead time")
    demand_std: float = Field(..., description="Demand standard deviation")
    stockout_probability: float = Field(..., description="Stockout probability")
    service_level: float = Field(..., description="Achieved service level")
