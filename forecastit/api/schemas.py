"""API request and response schemas."""

from datetime import date, datetime

from pydantic import BaseModel, Field, validator


class ForecastRequest(BaseModel):
    """Schema for forecast API requests."""

    store_id: str = Field(..., description="Store identifier")
    item_id: str = Field(..., description="Item identifier")
    start_date: date | datetime | str = Field(..., description="Forecast start date")
    end_date: date | datetime | str = Field(..., description="Forecast end date")
    promo_plan: list[int] | None = Field(None, description="Promo flags for forecast period")
    price_plan: list[float] | None = Field(None, description="Price plan for forecast period")

    @validator("start_date", "end_date", pre=True)
    def parse_date(cls, v):
        """Parse date from various formats."""
        if isinstance(v, str):
            try:
                import pandas as pd
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
    forecast_date: date | datetime | str = Field(..., description="Forecast date")
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
                import pandas as pd
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


class HealthResponse(BaseModel):
    """Schema for health check responses."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
