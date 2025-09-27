"""ForecastIt: Intelligent demand & sales forecasting system.

A production-ready forecasting system with cleaning pipelines, feature engineering,
multiple model families (classical/ML/DL), time-series CV & backtesting, explainability,
uncertainty quantification, API, dashboard, and orchestration.
"""

__version__ = "0.1.0"
__author__ = "ForecastIt Team"
__email__ = "team@forecastit.ai"

from forecastit.config.settings import Settings

# Initialize settings
settings = Settings()

__all__ = ["settings"]
