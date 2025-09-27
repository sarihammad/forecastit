"""Classical time series forecasting models."""

from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class SarimaxForecaster:
    """SARIMAX (Seasonal ARIMA with eXogenous variables) forecaster."""

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 12),
        trend: str | None = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ):
        """Initialize SARIMAX forecaster.

        Args:
            order: (p, d, q) order of the model.
            seasonal_order: (P, D, Q, s) seasonal order of the model.
            trend: Trend component ('n', 'c', 't', 'ct').
            enforce_stationarity: Whether to enforce stationarity.
            enforce_invertibility: Whether to enforce invertibility.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.logger = logger.bind(component="sarimax_forecaster")

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        exog_cols: list[str] | None = None,
        group_cols: list[str] | None = None,
    ) -> "SarimaxForecaster":
        """Fit SARIMAX model.

        Args:
            df: Training data.
            target_col: Name of target column.
            date_col: Name of date column.
            exog_cols: List of exogenous variable columns.
            group_cols: Columns to group by for multiple series.

        Returns:
            Fitted forecaster.
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            self.logger.info("Fitting SARIMAX model", order=self.order, seasonal_order=self.seasonal_order)

            # Prepare data
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values([date_col]).reset_index(drop=True)

            # For multiple series, fit separate models
            if group_cols and len(df.groupby(group_cols)) > 1:
                self.models = {}
                for group, group_data in df.groupby(group_cols):
                    self.models[group] = self._fit_single_series(
                        group_data, target_col, date_col, exog_cols
                    )
            else:
                self.model = self._fit_single_series(df, target_col, date_col, exog_cols)

            self.logger.info("SARIMAX model fitted successfully")
            return self

        except ImportError:
            raise ImportError("statsmodels is required for SARIMAX. Install with: pip install statsmodels")
        except Exception as e:
            self.logger.error(f"Failed to fit SARIMAX model: {e}")
            raise Exception(f"Failed to fit SARIMAX model: {e}") from e

    def _fit_single_series(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str,
        exog_cols: list[str] | None,
    ) -> Any:
        """Fit SARIMAX model for a single series.

        Args:
            df: Training data for single series.
            target_col: Name of target column.
            date_col: Name of date column.
            exog_cols: List of exogenous variable columns.

        Returns:
            Fitted SARIMAX model.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Prepare target series
        y = df[target_col].values

        # Prepare exogenous variables
        exog = None
        if exog_cols:
            exog = df[exog_cols].values

        # Fit model
        model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

        fitted_model = model.fit(disp=False)
        return fitted_model

    def predict(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Generate predictions.

        Args:
            horizon: Forecast horizon in periods.
            exog_future: Future exogenous variables.
            alpha: Significance level for confidence intervals.

        Returns:
            DataFrame with predictions and confidence intervals.
        """
        if self.model is None and not hasattr(self, "models"):
            raise ValueError("Model must be fitted before making predictions")

        self.logger.info("Generating SARIMAX predictions", horizon=horizon)

        try:
            if hasattr(self, "models"):
                # Multiple series case
                predictions = []
                for group, model in self.models.items():
                    group_pred = self._predict_single_series(
                        model, horizon, exog_future, alpha
                    )
                    # Add group information
                    for col in group:
                        group_pred[col] = group
                    predictions.append(group_pred)

                result = pd.concat(predictions, ignore_index=True)
            else:
                # Single series case
                result = self._predict_single_series(
                    self.model, horizon, exog_future, alpha
                )

            self.logger.info("SARIMAX predictions generated successfully")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate SARIMAX predictions: {e}")
            raise

    def _predict_single_series(
        self,
        model: Any,
        horizon: int,
        exog_future: pd.DataFrame | None,
        alpha: float,
    ) -> pd.DataFrame:
        """Generate predictions for a single series.

        Args:
            model: Fitted SARIMAX model.
            horizon: Forecast horizon.
            exog_future: Future exogenous variables.
            alpha: Significance level.

        Returns:
            DataFrame with predictions.
        """
        # Prepare future exogenous variables
        exog_future_values = None
        if exog_future is not None:
            exog_future_values = exog_future.values[:horizon]

        # Generate predictions with confidence intervals
        forecast = model.get_forecast(steps=horizon, exog=exog_future_values, alpha=alpha)

        # Extract results
        yhat = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Create result DataFrame
        result = pd.DataFrame({
            "yhat": yhat,
            "yhat_lower": conf_int.iloc[:, 0],
            "yhat_upper": conf_int.iloc[:, 1],
        })

        return result


class ProphetForecaster:
    """Prophet forecaster with external regressors."""

    def __init__(
        self,
        growth: str = "linear",
        seasonality_mode: str = "multiplicative",
        holidays_prior_scale: float = 10.0,
        seasonality_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
    ):
        """Initialize Prophet forecaster.

        Args:
            growth: Growth model ('linear', 'logistic').
            seasonality_mode: Seasonality mode ('additive', 'multiplicative').
            holidays_prior_scale: Prior scale for holidays.
            seasonality_prior_scale: Prior scale for seasonality.
            changepoint_prior_scale: Prior scale for changepoints.
        """
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self.logger = logger.bind(component="prophet_forecaster")

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        extra_regressors: list[str] | None = None,
        group_cols: list[str] | None = None,
    ) -> "ProphetForecaster":
        """Fit Prophet model.

        Args:
            df: Training data.
            target_col: Name of target column.
            date_col: Name of date column.
            extra_regressors: List of extra regressor columns.
            group_cols: Columns to group by for multiple series.

        Returns:
            Fitted forecaster.
        """
        try:
            from prophet import Prophet

            self.logger.info("Fitting Prophet model")

            # Prepare data
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values([date_col]).reset_index(drop=True)

            # For multiple series, fit separate models
            if group_cols and len(df.groupby(group_cols)) > 1:
                self.models = {}
                for group, group_data in df.groupby(group_cols):
                    self.models[group] = self._fit_single_series(
                        group_data, target_col, date_col, extra_regressors
                    )
            else:
                self.model = self._fit_single_series(df, target_col, date_col, extra_regressors)

            self.logger.info("Prophet model fitted successfully")
            return self

        except ImportError:
            raise ImportError("Prophet is required. Install with: pip install prophet")
        except Exception as e:
            self.logger.error(f"Failed to fit Prophet model: {e}")
            raise Exception(f"Failed to fit Prophet model: {e}") from e

    def _fit_single_series(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str,
        extra_regressors: list[str] | None,
    ) -> Any:
        """Fit Prophet model for a single series.

        Args:
            df: Training data for single series.
            target_col: Name of target column.
            date_col: Name of date column.
            extra_regressors: List of extra regressor columns.

        Returns:
            Fitted Prophet model.
        """
        from prophet import Prophet

        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            "ds": df[date_col],
            "y": df[target_col],
        })

        # Initialize Prophet model
        model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        # Add extra regressors
        if extra_regressors:
            for regressor in extra_regressors:
                if regressor in df.columns:
                    prophet_df[regressor] = df[regressor]
                    model.add_regressor(regressor)

        # Fit model
        fitted_model = model.fit(prophet_df)
        return fitted_model

    def predict(
        self,
        horizon: int,
        regressors_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate predictions.

        Args:
            horizon: Forecast horizon in periods.
            regressors_future: Future regressor values.

        Returns:
            DataFrame with predictions and confidence intervals.
        """
        if self.model is None and not hasattr(self, "models"):
            raise ValueError("Model must be fitted before making predictions")

        self.logger.info("Generating Prophet predictions", horizon=horizon)

        try:
            if hasattr(self, "models"):
                # Multiple series case
                predictions = []
                for group, model in self.models.items():
                    group_pred = self._predict_single_series(
                        model, horizon, regressors_future
                    )
                    # Add group information
                    for col in group:
                        group_pred[col] = group
                    predictions.append(group_pred)

                result = pd.concat(predictions, ignore_index=True)
            else:
                # Single series case
                result = self._predict_single_series(
                    self.model, horizon, regressors_future
                )

            self.logger.info("Prophet predictions generated successfully")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate Prophet predictions: {e}")
            raise

    def _predict_single_series(
        self,
        model: Any,
        horizon: int,
        regressors_future: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Generate predictions for a single series.

        Args:
            model: Fitted Prophet model.
            horizon: Forecast horizon.
            regressors_future: Future regressor values.

        Returns:
            DataFrame with predictions.
        """
        # Create future DataFrame
        future = model.make_future_dataframe(periods=horizon, freq="D")

        # Add future regressors
        if regressors_future is not None:
            for col in regressors_future.columns:
                if col in model.extra_regressors:
                    # Pad with last known values
                    last_values = future[col].dropna().iloc[-1:] if col in future.columns else [0]
                    future[col] = pd.concat([
                        future[col].fillna(method="ffill"),
                        pd.Series([last_values.iloc[0]] * horizon)
                    ]).values[:len(future)]

        # Generate forecast
        forecast = model.predict(future)

        # Extract only future predictions
        future_forecast = forecast.tail(horizon)

        # Create result DataFrame
        result = pd.DataFrame({
            "yhat": future_forecast["yhat"],
            "yhat_lower": future_forecast["yhat_lower"],
            "yhat_upper": future_forecast["yhat_upper"],
        })

        return result
