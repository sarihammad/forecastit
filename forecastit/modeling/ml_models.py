"""Machine learning forecasting models."""


import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class LGBMRegressorForecaster:
    """LightGBM regressor for point forecasting."""

    def __init__(
        self,
        objective: str = "regression",
        metric: str = "rmse",
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        verbose: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize LightGBM regressor.

        Args:
            objective: Objective function.
            metric: Evaluation metric.
            boosting_type: Boosting type.
            num_leaves: Number of leaves.
            learning_rate: Learning rate.
            feature_fraction: Feature fraction.
            bagging_fraction: Bagging fraction.
            bagging_freq: Bagging frequency.
            verbose: Verbosity level.
            random_state: Random state.
            **kwargs: Additional parameters.
        """
        self.params = {
            "objective": objective,
            "metric": metric,
            "boosting_type": boosting_type,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "verbose": verbose,
            "random_state": random_state,
            **kwargs,
        }
        self.model = None
        self.feature_names = None
        self.logger = logger.bind(component="lgbm_regressor")

    def fit(
        self,
        train_features: pd.DataFrame,
        y: pd.Series,
        categorical_features: list[str] | None = None,
        eval_set: list[tuple] | None = None,
        callbacks: list | None = None,
    ) -> "LGBMRegressorForecaster":
        """Fit LightGBM model.

        Args:
            train_features: Training features.
            y: Training targets.
            categorical_features: List of categorical feature names.
            eval_set: Evaluation set for early stopping.
            callbacks: Callbacks for training.

        Returns:
            Fitted forecaster.
        """
        try:
            import lightgbm as lgb

            self.logger.info("Fitting LightGBM regressor")

            # Store feature names
            self.feature_names = list(train_features.columns)

            # Handle categorical features
            if categorical_features is None:
                categorical_features = self._detect_categorical_features(train_features)

            # Convert categorical features to category type
            for col in categorical_features:
                if col in train_features.columns:
                    train_features[col] = train_features[col].astype("category")

            # Prepare data
            X = train_features.values

            # Create LightGBM dataset
            train_data = lgb.Dataset(
                X,
                label=y.values,
                feature_name=self.feature_names,
                categorical_feature=categorical_features,
            )

            # Fit model
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=eval_set,
                callbacks=callbacks,
            )

            self.logger.info("LightGBM regressor fitted successfully")
            return self

        except ImportError:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        except Exception as e:
            self.logger.error(f"Failed to fit LightGBM regressor: {e}")
            raise

    def predict(self, future_features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions.

        Args:
            future_features: Future feature values.

        Returns:
            DataFrame with predictions.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        self.logger.info("Generating LightGBM predictions")

        try:
            # Handle categorical features
            categorical_features = self._detect_categorical_features(future_features)
            for col in categorical_features:
                if col in future_features.columns:
                    future_features[col] = future_features[col].astype("category")

            # Ensure same feature order
            X = future_features[self.feature_names].values

            # Generate predictions
            yhat = self.model.predict(X)

            # Create result DataFrame
            result = pd.DataFrame({
                "yhat": yhat,
            })

            self.logger.info("LightGBM predictions generated successfully")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate LightGBM predictions: {e}")
            raise

    def _detect_categorical_features(self, df: pd.DataFrame) -> list[str]:
        """Detect categorical features.

        Args:
            df: DataFrame to analyze.

        Returns:
            List of categorical feature names.
        """
        categorical_features = []

        for col in df.columns:
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                categorical_features.append(col)
            elif col in ["store_id", "item_id"]:  # Common categorical features
                categorical_features.append(col)

        return categorical_features


class LGBMQuantileForecaster:
    """LightGBM quantile regressor for uncertainty quantification."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        verbose: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize LightGBM quantile forecaster.

        Args:
            quantiles: List of quantiles to predict.
            boosting_type: Boosting type.
            num_leaves: Number of leaves.
            learning_rate: Learning rate.
            feature_fraction: Feature fraction.
            bagging_fraction: Bagging fraction.
            bagging_freq: Bagging frequency.
            verbose: Verbosity level.
            random_state: Random state.
            **kwargs: Additional parameters.
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.base_params = {
            "boosting_type": boosting_type,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "verbose": verbose,
            "random_state": random_state,
            **kwargs,
        }
        self.models = {}
        self.feature_names = None
        self.logger = logger.bind(component="lgbm_quantile")

    def fit(
        self,
        train_features: pd.DataFrame,
        y: pd.Series,
        categorical_features: list[str] | None = None,
        eval_set: list[tuple] | None = None,
        callbacks: list | None = None,
    ) -> "LGBMQuantileForecaster":
        """Fit LightGBM quantile models.

        Args:
            train_features: Training features.
            y: Training targets.
            categorical_features: List of categorical feature names.
            eval_set: Evaluation set for early stopping.
            callbacks: Callbacks for training.

        Returns:
            Fitted forecaster.
        """
        try:
            import lightgbm as lgb

            self.logger.info("Fitting LightGBM quantile models", quantiles=self.quantiles)

            # Store feature names
            self.feature_names = list(train_features.columns)

            # Handle categorical features
            if categorical_features is None:
                categorical_features = self._detect_categorical_features(train_features)

            # Convert categorical features to category type
            for col in categorical_features:
                if col in train_features.columns:
                    train_features[col] = train_features[col].astype("category")

            # Prepare data
            X = train_features.values

            # Create LightGBM dataset
            train_data = lgb.Dataset(
                X,
                label=y.values,
                feature_name=self.feature_names,
                categorical_feature=categorical_features,
            )

            # Fit model for each quantile
            for quantile in self.quantiles:
                params = self.base_params.copy()
                params.update({
                    "objective": "quantile",
                    "alpha": quantile,
                    "metric": "quantile",
                })

                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=eval_set,
                    callbacks=callbacks,
                )

                self.models[quantile] = model

            self.logger.info("LightGBM quantile models fitted successfully")
            return self

        except ImportError:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        except Exception as e:
            self.logger.error(f"Failed to fit LightGBM quantile models: {e}")
            raise

    def predict(self, future_features: pd.DataFrame) -> pd.DataFrame:
        """Generate quantile predictions.

        Args:
            future_features: Future feature values.

        Returns:
            DataFrame with quantile predictions.
        """
        if not self.models:
            raise ValueError("Models must be fitted before making predictions")

        self.logger.info("Generating LightGBM quantile predictions")

        try:
            # Handle categorical features
            categorical_features = self._detect_categorical_features(future_features)
            for col in categorical_features:
                if col in future_features.columns:
                    future_features[col] = future_features[col].astype("category")

            # Ensure same feature order
            X = future_features[self.feature_names].values

            # Generate predictions for each quantile
            predictions = {}
            for quantile, model in self.models.items():
                predictions[f"q{int(quantile*100)}"] = model.predict(X)

            # Create result DataFrame
            result = pd.DataFrame(predictions)

            # Add standard column names
            if 0.5 in self.quantiles:
                result["yhat"] = result["q50"]
            if 0.1 in self.quantiles and 0.9 in self.quantiles:
                result["yhat_lower"] = result["q10"]
                result["yhat_upper"] = result["q90"]

            self.logger.info("LightGBM quantile predictions generated successfully")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate LightGBM quantile predictions: {e}")
            raise

    def _detect_categorical_features(self, df: pd.DataFrame) -> list[str]:
        """Detect categorical features.

        Args:
            df: DataFrame to analyze.

        Returns:
            List of categorical feature names.
        """
        categorical_features = []

        for col in df.columns:
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                categorical_features.append(col)
            elif col in ["store_id", "item_id"]:  # Common categorical features
                categorical_features.append(col)

        return categorical_features


class XGBRegressorForecaster:
    """XGBoost regressor for point forecasting."""

    def __init__(
        self,
        objective: str = "reg:squarederror",
        eval_metric: str = "rmse",
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize XGBoost regressor.

        Args:
            objective: Objective function.
            eval_metric: Evaluation metric.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate.
            n_estimators: Number of estimators.
            subsample: Subsample ratio.
            colsample_bytree: Column sample ratio.
            random_state: Random state.
            **kwargs: Additional parameters.
        """
        self.params = {
            "objective": objective,
            "eval_metric": eval_metric,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbosity": 0,
            **kwargs,
        }
        self.model = None
        self.feature_names = None
        self.logger = logger.bind(component="xgb_regressor")

    def fit(
        self,
        train_features: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "XGBRegressorForecaster":
        """Fit XGBoost model.

        Args:
            train_features: Training features.
            y: Training targets.
            eval_set: Evaluation set for early stopping.
            early_stopping_rounds: Early stopping rounds.

        Returns:
            Fitted forecaster.
        """
        try:
            import xgboost as xgb

            self.logger.info("Fitting XGBoost regressor")

            # Store feature names
            self.feature_names = list(train_features.columns)

            # Prepare data
            X = train_features.values

            # Fit model
            self.model = xgb.XGBRegressor(**self.params)

            if eval_set is not None:
                eval_set_xgb = [(X, y.values)] + [(eval_x, eval_y) for eval_x, eval_y in eval_set]
                self.model.fit(
                    X,
                    y.values,
                    eval_set=eval_set_xgb,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            else:
                self.model.fit(X, y.values)

            self.logger.info("XGBoost regressor fitted successfully")
            return self

        except ImportError:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        except Exception as e:
            self.logger.error(f"Failed to fit XGBoost regressor: {e}")
            raise

    def predict(self, future_features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions.

        Args:
            future_features: Future feature values.

        Returns:
            DataFrame with predictions.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        self.logger.info("Generating XGBoost predictions")

        try:
            # Ensure same feature order
            X = future_features[self.feature_names].values

            # Generate predictions
            yhat = self.model.predict(X)

            # Create result DataFrame
            result = pd.DataFrame({
                "yhat": yhat,
            })

            self.logger.info("XGBoost predictions generated successfully")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate XGBoost predictions: {e}")
            raise
