"""Plotting utilities for forecasting visualizations."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class ForecastingPlots:
    """Utilities for creating forecasting visualizations."""

    def __init__(self, output_dir: str = "reports/plots"):
        """Initialize plotting utilities.

        Args:
            output_dir: Directory to save plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="forecasting_plots")

    def plot_forecast_with_intervals(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        lower: pd.Series | None = None,
        upper: pd.Series | None = None,
        title: str = "Forecast vs Actual",
        save_path: str | None = None,
        show_plot: bool = False,
    ) -> str:
        """Plot forecast with confidence intervals.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            lower: Lower confidence bound.
            upper: Upper confidence bound.
            title: Plot title.
            save_path: Path to save plot.
            show_plot: Whether to display plot.

        Returns:
            Path to saved plot.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use("seaborn-v0_8")

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot actual
            ax.plot(actual.index, actual.values, label="Actual", linewidth=2, color="blue")

            # Plot forecast
            ax.plot(forecast.index, forecast.values, label="Forecast", linewidth=2, color="red", linestyle="--")

            # Plot confidence intervals
            if lower is not None and upper is not None:
                ax.fill_between(
                    forecast.index,
                    lower.values,
                    upper.values,
                    alpha=0.3,
                    color="red",
                    label="Confidence Interval",
                )

            # Customize plot
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            if save_path is None:
                save_path = self.output_dir / f"forecast_{title.replace(' ', '_').lower()}.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

            self.logger.info("Saved forecast plot", path=str(save_path))
            return str(save_path)

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping plot generation")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to create forecast plot: {e}")
            return ""

    def plot_residuals(
        self,
        actual: pd.Series,
        forecast: pd.Series,
        title: str = "Residuals Analysis",
        save_path: str | None = None,
        show_plot: bool = False,
    ) -> str:
        """Plot residual analysis.

        Args:
            actual: Actual values.
            forecast: Forecast values.
            title: Plot title.
            save_path: Path to save plot.
            show_plot: Whether to display plot.

        Returns:
            Path to saved plot.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use("seaborn-v0_8")

            # Calculate residuals
            residuals = actual - forecast

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Residuals over time
            axes[0, 0].plot(residuals.index, residuals.values)
            axes[0, 0].set_title("Residuals Over Time")
            axes[0, 0].set_xlabel("Date")
            axes[0, 0].set_ylabel("Residuals")
            axes[0, 0].grid(True, alpha=0.3)

            # Residuals histogram
            axes[0, 1].hist(residuals.values, bins=30, alpha=0.7)
            axes[0, 1].set_title("Residuals Distribution")
            axes[0, 1].set_xlabel("Residuals")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].grid(True, alpha=0.3)

            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals.values, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title("Q-Q Plot")
            axes[1, 0].grid(True, alpha=0.3)

            # Residuals vs fitted
            axes[1, 1].scatter(forecast.values, residuals.values, alpha=0.6)
            axes[1, 1].set_title("Residuals vs Fitted")
            axes[1, 1].set_xlabel("Fitted Values")
            axes[1, 1].set_ylabel("Residuals")
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle(title)
            plt.tight_layout()

            # Save plot
            if save_path is None:
                save_path = self.output_dir / f"residuals_{title.replace(' ', '_').lower()}.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

            self.logger.info("Saved residuals plot", path=str(save_path))
            return str(save_path)

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping residuals plot")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to create residuals plot: {e}")
            return ""

    def plot_feature_importance(
        self,
        feature_importance: dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 20,
        save_path: str | None = None,
        show_plot: bool = False,
    ) -> str:
        """Plot feature importance.

        Args:
            feature_importance: Dictionary with feature names and importance values.
            title: Plot title.
            top_n: Number of top features to show.
            save_path: Path to save plot.
            show_plot: Whether to display plot.

        Returns:
            Path to saved plot.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use("seaborn-v0_8")

            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:top_n]

            if not sorted_features:
                self.logger.warning("No feature importance data available")
                return ""

            # Extract names and values
            feature_names = [name for name, _ in sorted_features]
            importance_values = [value for _, value in sorted_features]

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Horizontal bar plot
            y_pos = np.arange(len(feature_names))
            bars = ax.barh(y_pos, importance_values)

            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel("Importance")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for _i, (bar, value) in enumerate(zip(bars, importance_values)):
                ax.text(
                    value + max(importance_values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    va="center",
                    ha="left",
                )

            plt.tight_layout()

            # Save plot
            if save_path is None:
                save_path = self.output_dir / f"feature_importance_{title.replace(' ', '_').lower()}.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

            self.logger.info("Saved feature importance plot", path=str(save_path))
            return str(save_path)

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping feature importance plot")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to create feature importance plot: {e}")
            return ""

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        title: str = "SHAP Summary",
        max_display: int = 20,
        save_path: str | None = None,
        show_plot: bool = False,
    ) -> str:
        """Plot SHAP summary.

        Args:
            shap_values: SHAP values array.
            feature_names: List of feature names.
            title: Plot title.
            max_display: Maximum number of features to display.
            save_path: Path to save plot.
            show_plot: Whether to display plot.

        Returns:
            Path to saved plot.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use("seaborn-v0_8")

            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)

            # Sort features by importance
            sorted_indices = np.argsort(mean_shap)[::-1][:max_display]

            # Extract top features
            top_features = [feature_names[i] for i in sorted_indices]
            top_values = mean_shap[sorted_indices]

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Horizontal bar plot
            y_pos = np.arange(len(top_features))
            bars = ax.barh(y_pos, top_values)

            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for _i, (bar, value) in enumerate(zip(bars, top_values)):
                ax.text(
                    value + max(top_values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    va="center",
                    ha="left",
                )

            plt.tight_layout()

            # Save plot
            if save_path is None:
                save_path = self.output_dir / f"shap_summary_{title.replace(' ', '_').lower()}.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

            self.logger.info("Saved SHAP summary plot", path=str(save_path))
            return str(save_path)

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping SHAP plot")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to create SHAP summary plot: {e}")
            return ""

    def plot_backtest_results(
        self,
        backtest_results: list[dict],
        model_name: str,
        metrics: list[str] | None = None,
        save_path: str | None = None,
        show_plot: bool = False,
    ) -> str:
        """Plot backtesting results across folds.

        Args:
            backtest_results: List of fold results.
            model_name: Name of the model.
            metrics: Metrics to plot.
            save_path: Path to save plot.
            show_plot: Whether to display plot.

        Returns:
            Path to saved plot.
        """
        if metrics is None:
            metrics = ["RMSE", "MAPE"]
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use("seaborn-v0_8")

            # Create subplots
            n_metrics = len(metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

            if n_metrics == 1:
                axes = [axes]

            # Plot each metric
            for i, metric in enumerate(metrics):
                fold_numbers = [result["fold"] for result in backtest_results]
                metric_values = [result["metrics"].get(metric, 0) for result in backtest_results]

                axes[i].bar(fold_numbers, metric_values)
                axes[i].set_title(f"{metric} by Fold")
                axes[i].set_xlabel("Fold")
                axes[i].set_ylabel(metric)
                axes[i].grid(True, alpha=0.3)

                # Add value labels on bars
                for _j, (fold, value) in enumerate(zip(fold_numbers, metric_values)):
                    axes[i].text(
                        fold,
                        value + max(metric_values) * 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

            plt.suptitle(f"{model_name} - Backtest Results")
            plt.tight_layout()

            # Save plot
            if save_path is None:
                save_path = self.output_dir / f"backtest_{model_name.replace(' ', '_').lower()}.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

            self.logger.info("Saved backtest results plot", path=str(save_path))
            return str(save_path)

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping backtest plot")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to create backtest results plot: {e}")
            return ""


def create_forecast_plot(
    actual: pd.Series,
    forecast: pd.Series,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    title: str = "Forecast vs Actual",
    save_path: str | None = None,
) -> str:
    """Convenience function to create forecast plot.

    Args:
        actual: Actual values.
        forecast: Forecast values.
        lower: Lower confidence bound.
        upper: Upper confidence bound.
        title: Plot title.
        save_path: Path to save plot.

    Returns:
        Path to saved plot.
    """
    plotter = ForecastingPlots()
    return plotter.plot_forecast_with_intervals(
        actual=actual,
        forecast=forecast,
        lower=lower,
        upper=upper,
        title=title,
        save_path=save_path,
    )


def create_residuals_plot(
    actual: pd.Series,
    forecast: pd.Series,
    title: str = "Residuals Analysis",
    save_path: str | None = None,
) -> str:
    """Convenience function to create residuals plot.

    Args:
        actual: Actual values.
        forecast: Forecast values.
        title: Plot title.
        save_path: Path to save plot.

    Returns:
        Path to saved plot.
    """
    plotter = ForecastingPlots()
    return plotter.plot_residuals(
        actual=actual,
        forecast=forecast,
        title=title,
        save_path=save_path,
    )


def create_feature_importance_plot(
    feature_importance: dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: str | None = None,
) -> str:
    """Convenience function to create feature importance plot.

    Args:
        feature_importance: Dictionary with feature names and importance values.
        title: Plot title.
        top_n: Number of top features to show.
        save_path: Path to save plot.

    Returns:
        Path to saved plot.
    """
    plotter = ForecastingPlots()
    return plotter.plot_feature_importance(
        feature_importance=feature_importance,
        title=title,
        top_n=top_n,
        save_path=save_path,
    )
