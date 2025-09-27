"""Model registry and MLflow integration."""

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import structlog

from forecastit.config.settings import Settings

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Model registry using MLflow."""

    def __init__(self, settings: Settings):
        """Initialize model registry.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.logger = logger.bind(component="model_registry")

        # Set up MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.experiment_name = "forecastit"
        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Set up MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                self.logger.info("Created new MLflow experiment", experiment_id=experiment_id)
            else:
                experiment_id = experiment.experiment_id
                self.logger.info("Using existing MLflow experiment", experiment_id=experiment_id)
        except Exception as e:
            self.logger.error("Failed to set up MLflow experiment", error=str(e))
            raise

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.

        Args:
            run_name: Name for the run.
            tags: Tags for the run.

        Returns:
            Active MLflow run.
        """
        if tags is None:
            tags = {}

        run = mlflow.start_run(
            experiment_id=mlflow.get_experiment_by_name(self.experiment_name).experiment_id,
            run_name=run_name,
            tags=tags,
        )

        self.logger.info("Started MLflow run", run_id=run.info.run_id, run_name=run_name)

        return run

    def log_model_parameters(self, params: dict[str, Any]) -> None:
        """Log model parameters.

        Args:
            params: Dictionary of parameters to log.
        """
        mlflow.log_params(params)
        self.logger.info("Logged model parameters", param_count=len(params))

    def log_model_metrics(self, metrics: dict[str, float]) -> None:
        """Log model metrics.

        Args:
            metrics: Dictionary of metrics to log.
        """
        mlflow.log_metrics(metrics)
        self.logger.info("Logged model metrics", metric_count=len(metrics))

    def log_model_artifact(self, artifact_path: str | Path, artifact_name: str) -> None:
        """Log a model artifact.

        Args:
            artifact_path: Path to the artifact.
            artifact_name: Name for the artifact.
        """
        artifact_path = Path(artifact_path)
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_name)
            self.logger.info("Logged model artifact", artifact_name=artifact_name)
        else:
            self.logger.warning("Artifact not found", artifact_path=str(artifact_path))

    def log_model(
        self,
        model: Any,
        model_name: str,
        signature: mlflow.models.signature.ModelSignature | None = None,
        input_example: pd.DataFrame | None = None,
        conda_env: dict | None = None,
    ) -> str:
        """Log a trained model.

        Args:
            model: Trained model object.
            model_name: Name for the model.
            signature: Model signature.
            input_example: Example input data.
            conda_env: Conda environment specification.

        Returns:
            Model URI.
        """
        # Determine model flavor
        model_type = type(model).__name__.lower()

        if "lightgbm" in model_type:
            model_uri = mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                conda_env=conda_env,
            )
        elif "xgboost" in model_type:
            model_uri = mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                conda_env=conda_env,
            )
        elif "sklearn" in model_type or hasattr(model, "predict"):
            model_uri = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                conda_env=conda_env,
            )
        else:
            # Generic model logging
            model_uri = mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=model,
                signature=signature,
                input_example=input_example,
                conda_env=conda_env,
            )

        self.logger.info("Logged model", model_name=model_name, model_uri=model_uri)

        return model_uri

    def load_model(self, model_uri: str) -> Any:
        """Load a model from MLflow.

        Args:
            model_uri: URI of the model to load.

        Returns:
            Loaded model.
        """
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            self.logger.info("Loaded model", model_uri=model_uri)
            return model
        except Exception as e:
            self.logger.error("Failed to load model", model_uri=model_uri, error=str(e))
            raise

    def get_best_model(
        self,
        metric: str = "RMSE",
        ascending: bool = True,
        limit: int = 1,
    ) -> dict | None:
        """Get the best model based on a metric.

        Args:
            metric: Metric to use for ranking.
            ascending: Whether to sort in ascending order.
            limit: Number of results to return.

        Returns:
            Dictionary with best model information.
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.logger.warning("Experiment not found")
                return None

            # Get all runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="",
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                max_results=1000,
            )

            if runs.empty:
                self.logger.warning("No runs found")
                return None

            # Filter runs with the metric
            metric_col = f"metrics.{metric}"
            if metric_col not in runs.columns:
                self.logger.warning(f"Metric {metric} not found in runs")
                return None

            # Remove runs without the metric
            runs_with_metric = runs.dropna(subset=[metric_col])

            if runs_with_metric.empty:
                self.logger.warning(f"No runs with metric {metric}")
                return None

            # Sort by metric
            sorted_runs = runs_with_metric.sort_values(metric_col, ascending=ascending)

            best_runs = sorted_runs.head(limit)

            results = []
            for _, run in best_runs.iterrows():
                result = {
                    "run_id": run["run_id"],
                    "run_name": run.get("tags.mlflow.runName", ""),
                    "metric_value": run[metric_col],
                    "model_uri": run.get("artifact_uri", ""),
                    "start_time": run["start_time"],
                }
                results.append(result)

            self.logger.info(
                "Retrieved best models",
                metric=metric,
                count=len(results),
                best_value=results[0]["metric_value"] if results else None,
            )

            return results[0] if limit == 1 else results

        except Exception as e:
            self.logger.error("Failed to get best model", error=str(e))
            return None

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        version: str | None = None,
        stage: str = "None",
        description: str | None = None,
    ) -> str:
        """Register a model in the model registry.

        Args:
            model_uri: URI of the model to register.
            model_name: Name for the registered model.
            version: Version for the model.
            stage: Stage for the model.
            description: Description for the model.

        Returns:
            Model version.
        """
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={"description": description} if description else None,
            )

            # Transition to stage if specified
            if stage != "None":
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage=stage,
                )

            self.logger.info(
                "Registered model",
                model_name=model_name,
                version=model_version.version,
                stage=stage,
            )

            return model_version.version

        except Exception as e:
            self.logger.error("Failed to register model", error=str(e))
            raise

    def get_registered_model(self, model_name: str, stage: str = "Production") -> str | None:
        """Get a registered model URI.

        Args:
            model_name: Name of the registered model.
            stage: Stage of the model.

        Returns:
            Model URI if found.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(
                name=model_name,
                stages=[stage],
            )

            if model_version:
                model_uri = f"models:/{model_name}/{stage}"
                self.logger.info("Retrieved registered model", model_name=model_name, stage=stage)
                return model_uri
            else:
                self.logger.warning("No model found", model_name=model_name, stage=stage)
                return None

        except Exception as e:
            self.logger.error("Failed to get registered model", error=str(e))
            return None

    def log_data_info(self, data: pd.DataFrame, data_name: str) -> None:
        """Log information about a dataset.

        Args:
            data: DataFrame to log info about.
            data_name: Name for the dataset.
        """
        info = {
            "shape": data.shape,
            "columns": list(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": data.isnull().sum().to_dict(),
        }

        # Log as parameters
        for key, value in info.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"{data_name}_{key}_{sub_key}", sub_value)
            else:
                mlflow.log_param(f"{data_name}_{key}", value)

        self.logger.info("Logged data info", data_name=data_name)

    def log_feature_importance(
        self,
        feature_names: list[str],
        importance_values: np.ndarray,
        model_name: str,
    ) -> None:
        """Log feature importance.

        Args:
            feature_names: List of feature names.
            importance_values: Feature importance values.
            model_name: Name of the model.
        """
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values,
        }).sort_values("importance", ascending=False)

        # Log top features as parameters
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            mlflow.log_param(f"{model_name}_top_feature_{i+1}", row["feature"])
            mlflow.log_param(f"{model_name}_top_importance_{i+1}", row["importance"])

        # Save feature importance as artifact
        importance_path = f"{model_name}_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # Clean up
        os.remove(importance_path)

        self.logger.info("Logged feature importance", model_name=model_name)

    def load_champion(self, model_name: str = "forecastit_champion") -> tuple:
        """Load champion model and preprocessor from registry.

        Args:
            model_name: Name of the champion model.

        Returns:
            Tuple of (model, preprocessor).
        """
        try:
            # Try to load from model registry first
            model_uri = self.get_registered_model(model_name, "Production")

            if model_uri:
                # Load model
                model = self.load_model(model_uri)

                # Try to load preprocessor from artifacts
                try:
                    preprocessor = mlflow.pyfunc.load_model(f"{model_uri}_preprocessor")
                    self.logger.info("Loaded champion model and preprocessor from registry")
                    return model, preprocessor
                except Exception:
                    # Preprocessor not found, return model with None preprocessor
                    self.logger.warning("Champion model loaded but preprocessor not found")
                    return model, None
            else:
                # Fallback to best model from experiments
                self.logger.info("Champion model not found in registry, falling back to best model")
                best_model = self.get_best_model(metric="sMAPE", ascending=True)

                if best_model:
                    model = self.load_model(best_model["model_uri"])
                    return model, None
                else:
                    raise ValueError("No champion model found")

        except Exception as e:
            self.logger.error("Failed to load champion model", error=str(e))
            raise

    def register_model_with_metadata(
        self,
        model_uri: str,
        model_name: str,
        description: str,
        stage: str = "Production",
        signature: mlflow.models.signature.ModelSignature | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register model with comprehensive metadata.

        Args:
            model_uri: URI of the model to register.
            model_name: Name for the registered model.
            description: Description for the model.
            stage: Stage for the model.
            signature: Model signature.
            metadata: Additional metadata.

        Returns:
            Model version.
        """
        try:
            # Register model
            version = self.register_model(
                model_uri=model_uri,
                model_name=model_name,
                stage=stage,
                description=description,
            )

            # Add metadata as tags
            if metadata:
                client = mlflow.tracking.MlflowClient()
                for key, value in metadata.items():
                    client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=key,
                        value=str(value),
                    )

            self.logger.info("Registered model with metadata", model_name=model_name, version=version)
            return version

        except Exception as e:
            self.logger.error("Failed to register model with metadata", error=str(e))
            raise

    def list_registered_models(self) -> list[dict[str, Any]]:
        """List all registered models.

        Returns:
            List of registered model information.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            registered_models = client.search_registered_models()

            models = []
            for model in registered_models:
                latest_version = client.get_latest_versions(model.name, stages=["Production"])
                model_info = {
                    "name": model.name,
                    "latest_version": latest_version[0].version if latest_version else None,
                    "stages": [version.current_stage for version in model.latest_versions],
                    "creation_timestamp": model.creation_timestamp,
                    "description": model.description,
                }
                models.append(model_info)

            self.logger.info("Listed registered models", count=len(models))
            return models

        except Exception as e:
            self.logger.error("Failed to list registered models", error=str(e))
            return []

    def promote_model(
        self,
        model_name: str,
        version: str,
        from_stage: str,
        to_stage: str,
        comment: str | None = None,
    ) -> None:
        """Promote a model to a different stage.

        Args:
            model_name: Name of the model.
            version: Version of the model.
            from_stage: Current stage.
            to_stage: Target stage.
            comment: Optional comment for the transition.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=to_stage,
                archive_existing_versions=False,
            )

            if comment:
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="transition_comment",
                    value=comment,
                )

            self.logger.info(
                "Promoted model",
                model_name=model_name,
                version=version,
                from_stage=from_stage,
                to_stage=to_stage,
            )

        except Exception as e:
            self.logger.error("Failed to promote model", error=str(e))
            raise

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        self.logger.info("Ended MLflow run")
