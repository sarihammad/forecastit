"""Application settings and configuration management."""

from pathlib import Path

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Environment
    environment: str = Field(default="development", env="FORECASTIT_ENV")

    # Data paths
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    reports_dir: Path = Field(default=Path("./reports"), env="REPORTS_DIR")

    # MLflow
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db", env="MLFLOW_TRACKING_URI")
    mlflow_artifact_root: str = Field(default="./mlruns", env="MLFLOW_ARTIFACT_ROOT")

    # Kaggle API (optional)
    kaggle_username: str | None = Field(default=None, env="KAGGLE_USERNAME")
    kaggle_key: str | None = Field(default=None, env="KAGGLE_KEY")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    # Streamlit Configuration
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    streamlit_host: str = Field(default="0.0.0.0", env="STREAMLIT_HOST")

    # Model Configuration
    default_horizon: int = Field(default=28, env="DEFAULT_HORIZON")
    default_folds: int = Field(default=6, env="DEFAULT_FOLDS")
    default_metric: str = Field(default="RMSE", env="DEFAULT_METRIC")

    # Random Seeds
    random_seed: int = Field(default=42, env="RANDOM_SEED")
    np_random_seed: int = Field(default=42, env="NP_RANDOM_SEED")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    # Feature Engineering
    max_lag: int = Field(default=28, env="MAX_LAG")
    rolling_windows: list[int] = Field(default=[7, 14, 28], env="ROLLING_WINDOWS")
    ema_alpha: float = Field(default=0.3, env="EMA_ALPHA")

    # Model Training
    optuna_n_trials: int = Field(default=100, env="OPTUNA_N_TRIALS")
    optuna_timeout: int = Field(default=3600, env="OPTUNA_TIMEOUT")
    backtest_horizon: int = Field(default=28, env="BACKTEST_HORIZON")
    backtest_folds: int = Field(default=6, env="BACKTEST_FOLDS")

    # API Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")

    # Inventory Management
    default_service_level: float = Field(default=0.95, env="DEFAULT_SERVICE_LEVEL")
    default_lead_time: int = Field(default=7, env="DEFAULT_LEAD_TIME")
    default_holding_cost: float = Field(default=0.1, env="DEFAULT_HOLDING_COST")
    default_stockout_cost: float = Field(default=10.0, env="DEFAULT_STOCKOUT_COST")

    # Model Families Configuration
    enable_classical: bool = Field(default=True)
    enable_ml: bool = Field(default=True)
    enable_dl: bool = Field(default=False)
    enable_prophet: bool = Field(default=True, env="FORECASTIT_ENABLE_PROPHET")
    disable_prophet: bool = Field(default=False, env="FORECASTIT_DISABLE_PROPHET")
    enable_shap: bool = Field(default=True, env="FORECASTIT_ENABLE_SHAP")

    # Feature Engineering Flags
    include_calendar_features: bool = Field(default=True)
    include_lag_features: bool = Field(default=True)
    include_rolling_features: bool = Field(default=True)
    include_price_promo_features: bool = Field(default=True)
    include_weather_features: bool = Field(default=False)

    # Model Selection
    model_selection_strategy: str = Field(default="best_overall")  # best_overall, best_per_segment

    @validator("data_dir", "reports_dir", pre=True)
    def create_directories(cls, v: str) -> Path:
        """Ensure directories exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @validator("rolling_windows", pre=True)
    def parse_rolling_windows(cls, v: str) -> list[int]:
        """Parse rolling windows from string or return as-is if list."""
        if isinstance(v, str):
            # Parse from string like "[7, 14, 28]"
            import ast
            return ast.literal_eval(v)
        return v

    @property
    def raw_data_dir(self) -> Path:
        """Raw data directory."""
        return self.data_dir / "raw"

    @property
    def interim_data_dir(self) -> Path:
        """Interim (cleaned) data directory."""
        return self.data_dir / "interim"

    @property
    def features_data_dir(self) -> Path:
        """Features data directory."""
        return self.data_dir / "features"

    @property
    def models_dir(self) -> Path:
        """Models directory."""
        return self.data_dir / "models"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
