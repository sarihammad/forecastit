"""FastAPI application for ForecastIt."""

import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from forecastit.api.schemas import (
    ErrorResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    InventoryRequest,
    InventoryResponse,
)
from forecastit.config.settings import Settings
from forecastit.inference.inventory import InventoryOptimizer
from forecastit.inference.predictor import Predictor
from forecastit.utils.logging import setup_logging

# Initialize settings and logging
settings = Settings()
setup_logging(settings)
logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ForecastIt API",
    description="Intelligent demand & sales forecasting system",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
predictor: Predictor | None = None
inventory_optimizer: InventoryOptimizer | None = None

# Rate limiting (simple in-memory token bucket)
rate_limit_tokens = defaultdict(lambda: {"tokens": 100, "last_update": time.time()})
RATE_LIMIT_MAX = 100
RATE_LIMIT_REFILL_RATE = 10  # tokens per second


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global predictor, inventory_optimizer

    logger.info("Starting ForecastIt API")

    try:
        # Initialize predictor
        predictor = Predictor(settings)
        await predictor.load_model()

        # Initialize inventory optimizer
        inventory_optimizer = InventoryOptimizer(settings)

        logger.info("API startup completed successfully")

    except Exception as e:
        logger.error("Failed to initialize API", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ForecastIt API")


def get_predictor() -> Predictor:
    """Dependency to get predictor instance."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    return predictor


def get_inventory_optimizer() -> InventoryOptimizer:
    """Dependency to get inventory optimizer instance."""
    if inventory_optimizer is None:
        raise HTTPException(status_code=503, detail="Inventory optimizer not initialized")
    return inventory_optimizer


def check_rate_limit(client_ip: str) -> bool:
    """Check if client is within rate limit.

    Args:
        client_ip: Client IP address.

    Returns:
        True if within limit, False otherwise.
    """
    current_time = time.time()
    client_data = rate_limit_tokens[client_ip]

    # Refill tokens based on time elapsed
    time_elapsed = current_time - client_data["last_update"]
    tokens_to_add = time_elapsed * RATE_LIMIT_REFILL_RATE
    client_data["tokens"] = min(RATE_LIMIT_MAX, client_data["tokens"] + tokens_to_add)
    client_data["last_update"] = current_time

    # Check if request is allowed
    if client_data["tokens"] >= 1:
        client_data["tokens"] -= 1
        return True

    return False


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_ip = request.client.host

    if not check_rate_limit(client_ip):
        logger.warning("Rate limit exceeded", client_ip=client_ip)
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": "Too many requests"}
        )

    response = await call_next(request)
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=predictor is not None,
    )


@app.post("/predict", response_model=list[ForecastResponse])
async def predict(
    request: ForecastRequest,
    predictor_instance: Predictor = Depends(get_predictor),
):
    """Generate forecasts for a specific store-item combination."""
    try:
        # Validate input ranges
        if request.promo_plan:
            for promo in request.promo_plan:
                if not isinstance(promo, int) or promo < 0 or promo > 1:
                    raise HTTPException(status_code=400, detail="Promo values must be 0 or 1")

        if request.price_plan:
            for price in request.price_plan:
                if price <= 0:
                    raise HTTPException(status_code=400, detail="Price values must be positive")

        # Check date range
        date_diff = (request.end_date - request.start_date).days
        if date_diff < 0:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        if date_diff > 365:
            raise HTTPException(status_code=400, detail="Forecast horizon cannot exceed 365 days")

        logger.info(
            "Received forecast request",
            store_id=request.store_id,
            item_id=request.item_id,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        # Generate forecasts
        forecasts = await predictor_instance.predict(
            store_id=request.store_id,
            item_id=request.item_id,
            start_date=request.start_date,
            end_date=request.end_date,
            promo_plan=request.promo_plan,
            price_plan=request.price_plan,
        )

        # Convert to response format
        responses = []
        for forecast in forecasts:
            response = ForecastResponse(
                store_id=forecast["store_id"],
                item_id=forecast["item_id"],
                forecast_date=forecast["date"],
                yhat=forecast["yhat"],
                yhat_lower=forecast["yhat_lower"],
                yhat_upper=forecast["yhat_upper"],
                model_name=forecast["model_name"],
                model_version=forecast["model_version"],
            )
            responses.append(response)

        logger.info(
            "Generated forecasts",
            store_id=request.store_id,
            item_id=request.item_id,
            forecast_count=len(responses),
        )

        return responses

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Forecast generation failed",
            store_id=request.store_id,
            item_id=request.item_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {e!s}") from e


@app.post("/inventory/suggest", response_model=InventoryResponse)
async def suggest_inventory(
    request: InventoryRequest,
    inventory_optimizer_instance: InventoryOptimizer = Depends(get_inventory_optimizer),
    predictor_instance: Predictor = Depends(get_predictor),
):
    """Generate inventory optimization suggestions."""
    try:
        # Validate input ranges
        if not 0.5 <= request.service_level <= 0.999:
            raise HTTPException(status_code=400, detail="Service level must be between 0.5 and 0.999")

        if request.lead_time <= 0 or request.lead_time > 365:
            raise HTTPException(status_code=400, detail="Lead time must be between 1 and 365 days")

        if request.holding_cost < 0:
            raise HTTPException(status_code=400, detail="Holding cost must be non-negative")

        if request.stockout_cost < 0:
            raise HTTPException(status_code=400, detail="Stockout cost must be non-negative")

        logger.info(
            "Received inventory request",
            store_id=request.store_id,
            item_id=request.item_id,
            service_level=request.service_level,
            lead_time=request.lead_time,
        )

        # Generate forecasts for lead time period
        from datetime import timedelta
        end_date = datetime.now().date() + timedelta(days=request.lead_time)

        forecasts = await predictor_instance.predict(
            store_id=request.store_id,
            item_id=request.item_id,
            start_date=datetime.now().date(),
            end_date=end_date,
        )

        # Calculate inventory metrics
        inventory_metrics = inventory_optimizer_instance.calculate_inventory_metrics(
            forecasts=forecasts,
            service_level=request.service_level,
            lead_time=request.lead_time,
            holding_cost=request.holding_cost,
            stockout_cost=request.stockout_cost,
        )

        response = InventoryResponse(
            store_id=request.store_id,
            item_id=request.item_id,
            reorder_point=inventory_metrics["reorder_point"],
            safety_stock=inventory_metrics["safety_stock"],
            expected_demand=inventory_metrics["expected_demand"],
            demand_std=inventory_metrics["demand_std"],
            stockout_probability=inventory_metrics["stockout_probability"],
            service_level=inventory_metrics["service_level"],
        )

        logger.info(
            "Generated inventory suggestions",
            store_id=request.store_id,
            item_id=request.item_id,
            reorder_point=response.reorder_point,
            safety_stock=response.safety_stock,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Inventory optimization failed",
            store_id=request.store_id,
            item_id=request.item_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Inventory optimization failed: {e!s}") from e


@app.get("/models")
async def list_models():
    """List available models."""
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Predictor not initialized")

        models = await predictor.list_available_models()

        return {
            "models": models,
            "current_model": predictor.current_model_info,
        }

    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e!s}") from e


@app.get("/explain/{store_id}/{item_id}")
async def explain_forecast(
    store_id: str,
    item_id: str,
    predictor_instance: Predictor = Depends(get_predictor),
):
    """Get feature importance explanation for a forecast."""
    try:
        logger.info(
            "Received explanation request",
            store_id=store_id,
            item_id=item_id,
        )

        # Get feature importance
        explanation = predictor_instance.explain_prediction(
            store_id=store_id,
            item_id=item_id,
            forecast_date=pd.Timestamp.now(),
            top_k=10,
        )

        return explanation

    except Exception as e:
        logger.error(
            "Feature explanation failed",
            store_id=store_id,
            item_id=item_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Feature explanation failed: {e!s}") from e


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "forecastit.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
    )
