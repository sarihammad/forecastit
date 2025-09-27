"""Data quality reporting and profiling."""

from pathlib import Path

import pandas as pd
import structlog

from forecastit.config.settings import Settings

logger = structlog.get_logger(__name__)


class QualityReporter:
    """Generate data quality reports and profiles."""

    def __init__(self, settings: Settings):
        """Initialize quality reporter.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.logger = logger.bind(component="quality_reporter")

    def generate_report(
        self,
        data: pd.DataFrame,
        output_dir: Path | None = None,
        include_plots: bool = True,
    ) -> dict:
        """Generate comprehensive data quality report.

        Args:
            data: DataFrame to analyze.
            output_dir: Directory to save reports (defaults to reports_dir).
            include_plots: Whether to include visualizations.

        Returns:
            Dictionary with quality metrics.
        """
        if output_dir is None:
            output_dir = self.settings.reports_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating quality report", shape=data.shape, output_dir=str(output_dir))

        # Generate quality metrics
        quality_metrics = self._analyze_data_quality(data)

        # Generate missingness analysis
        missingness = self._analyze_missingness(data)

        # Generate duplicate analysis
        duplicates = self._analyze_duplicates(data)

        # Generate outlier analysis
        outliers = self._analyze_outliers(data)

        # Generate time series analysis
        ts_analysis = self._analyze_time_series(data)

        # Combine all analyses
        report = {
            "data_overview": quality_metrics,
            "missingness": missingness,
            "duplicates": duplicates,
            "outliers": outliers,
            "time_series": ts_analysis,
        }

        # Save reports
        self._save_html_report(report, output_dir / "quality_report.html")
        self._save_markdown_report(report, output_dir / "quality_report.md")
        self._save_json_report(report, output_dir / "quality_metrics.json")

        if include_plots:
            self._generate_plots(data, report, output_dir)

        self.logger.info("Quality report generated", output_dir=str(output_dir))

        return report

    def _analyze_data_quality(self, data: pd.DataFrame) -> dict:
        """Analyze basic data quality metrics.

        Args:
            data: DataFrame to analyze.

        Returns:
            Dictionary with quality metrics.
        """
        return {
            "shape": data.shape,
            "memory_usage": data.memory_usage(deep=True).sum(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "unique_values": {col: data[col].nunique() for col in data.columns},
            "date_range": {
                "start": data["date"].min() if "date" in data.columns else None,
                "end": data["date"].max() if "date" in data.columns else None,
            },
        }

    def _analyze_missingness(self, data: pd.DataFrame) -> dict:
        """Analyze missing values.

        Args:
            data: DataFrame to analyze.

        Returns:
            Dictionary with missingness analysis.
        """
        missing_count = data.isnull().sum()
        missing_pct = (missing_count / len(data)) * 100

        # Identify columns with missing values
        columns_with_missing = missing_count[missing_count > 0].to_dict()
        columns_with_missing_pct = missing_pct[missing_count > 0].to_dict()

        # Analyze patterns of missingness
        missing_patterns = {}
        for col in data.columns:
            if missing_count[col] > 0:
                # Check if missing values are clustered
                is_missing = data[col].isnull()
                if is_missing.sum() > 0:
                    # Find consecutive missing values
                    consecutive_missing = []
                    current_length = 0
                    for val in is_missing:
                        if val:
                            current_length += 1
                        else:
                            if current_length > 0:
                                consecutive_missing.append(current_length)
                                current_length = 0
                    if current_length > 0:
                        consecutive_missing.append(current_length)

                    missing_patterns[col] = {
                        "total_missing": missing_count[col],
                        "missing_percentage": missing_pct[col],
                        "max_consecutive": max(consecutive_missing) if consecutive_missing else 0,
                        "avg_consecutive": sum(consecutive_missing) / len(consecutive_missing) if consecutive_missing else 0,
                    }

        return {
            "total_missing": missing_count.sum(),
            "missing_percentage": (missing_count.sum() / (len(data) * len(data.columns))) * 100,
            "columns_with_missing": columns_with_missing,
            "columns_with_missing_pct": columns_with_missing_pct,
            "missing_patterns": missing_patterns,
        }

    def _analyze_duplicates(self, data: pd.DataFrame) -> dict:
        """Analyze duplicate records.

        Args:
            data: DataFrame to analyze.

        Returns:
            Dictionary with duplicate analysis.
        """
        # Exact duplicates
        exact_duplicates = data.duplicated().sum()

        # Duplicates by key columns (if they exist)
        key_duplicates = {}
        if "date" in data.columns and "store_id" in data.columns and "item_id" in data.columns:
            key_duplicates["by_key"] = data.duplicated(subset=["date", "store_id", "item_id"]).sum()

        # Duplicates by store-item
        if "store_id" in data.columns and "item_id" in data.columns:
            key_duplicates["by_store_item"] = data.duplicated(subset=["store_id", "item_id"]).sum()

        return {
            "exact_duplicates": exact_duplicates,
            "exact_duplicates_pct": (exact_duplicates / len(data)) * 100,
            "key_duplicates": key_duplicates,
        }

    def _analyze_outliers(self, data: pd.DataFrame) -> dict:
        """Analyze outliers in numerical columns.

        Args:
            data: DataFrame to analyze.

        Returns:
            Dictionary with outlier analysis.
        """
        numerical_columns = data.select_dtypes(include=["number"]).columns
        outliers = {}

        for col in numerical_columns:
            if col in data.columns:
                # IQR method
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outlier_count = outlier_mask.sum()

                # Z-score method
                z_scores = abs((data[col] - data[col].mean()) / data[col].std())
                z_outlier_count = (z_scores > 3).sum()

                outliers[col] = {
                    "iqr_outliers": {
                        "count": outlier_count,
                        "percentage": (outlier_count / len(data)) * 100,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    },
                    "z_score_outliers": {
                        "count": z_outlier_count,
                        "percentage": (z_outlier_count / len(data)) * 100,
                    },
                    "statistics": {
                        "mean": data[col].mean(),
                        "median": data[col].median(),
                        "std": data[col].std(),
                        "min": data[col].min(),
                        "max": data[col].max(),
                    },
                }

        return outliers

    def _analyze_time_series(self, data: pd.DataFrame) -> dict:
        """Analyze time series properties.

        Args:
            data: DataFrame to analyze.

        Returns:
            Dictionary with time series analysis.
        """
        if "date" not in data.columns:
            return {"error": "No date column found"}

        # Convert date to datetime
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])

        # Check for missing dates
        date_range = pd.date_range(start=data["date"].min(), end=data["date"].max(), freq="D")
        missing_dates = set(date_range) - set(data["date"])

        # Analyze by store and item
        store_item_analysis = {}
        if "store_id" in data.columns and "item_id" in data.columns:
            for store in data["store_id"].unique():
                for item in data["item_id"].unique():
                    subset = data[(data["store_id"] == store) & (data["item_id"] == item)]
                    if len(subset) > 0:
                        subset_dates = pd.date_range(
                            start=subset["date"].min(),
                            end=subset["date"].max(),
                            freq="D"
                        )
                        subset_missing = set(subset_dates) - set(subset["date"])

                        store_item_analysis[f"{store}_{item}"] = {
                            "total_days": len(subset),
                            "missing_days": len(subset_missing),
                            "completeness": (len(subset) / len(subset_dates)) * 100,
                        }

        return {
            "total_days": len(data),
            "date_range": {
                "start": data["date"].min(),
                "end": data["date"].max(),
                "span_days": (data["date"].max() - data["date"].min()).days,
            },
            "missing_dates": {
                "count": len(missing_dates),
                "dates": sorted(missing_dates)[:10],  # First 10 missing dates
            },
            "store_item_analysis": store_item_analysis,
        }

    def _save_html_report(self, report: dict, output_path: Path) -> None:
        """Save HTML quality report.

        Args:
            report: Quality report data.
            output_path: Path to save HTML report.
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .metric {{ margin: 10px 0; }}
                .metric-label {{ font-weight: bold; }}
                .metric-value {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .warning {{ color: #ff6600; }}
                .error {{ color: #cc0000; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>

            <h2>Data Overview</h2>
            <div class="metric">
                <span class="metric-label">Shape:</span>
                <span class="metric-value">{report['data_overview']['shape']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Memory Usage:</span>
                <span class="metric-value">{report['data_overview']['memory_usage_mb']:.2f} MB</span>
            </div>

            <h2>Missing Values</h2>
            <div class="metric">
                <span class="metric-label">Total Missing:</span>
                <span class="metric-value">{report['missingness']['total_missing']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Missing Percentage:</span>
                <span class="metric-value">{report['missingness']['missing_percentage']:.2f}%</span>
            </div>

            <h2>Duplicates</h2>
            <div class="metric">
                <span class="metric-label">Exact Duplicates:</span>
                <span class="metric-value">{report['duplicates']['exact_duplicates']}</span>
            </div>

            <h2>Time Series Analysis</h2>
            <div class="metric">
                <span class="metric-label">Date Range:</span>
                <span class="metric-value">{report['time_series']['date_range']['start']} to {report['time_series']['date_range']['end']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Missing Dates:</span>
                <span class="metric-value">{report['time_series']['missing_dates']['count']}</span>
            </div>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

        self.logger.info("Saved HTML quality report", path=str(output_path))

    def _save_markdown_report(self, report: dict, output_path: Path) -> None:
        """Save Markdown quality report.

        Args:
            report: Quality report data.
            output_path: Path to save Markdown report.
        """
        md_content = f"""# Data Quality Report

## Data Overview
- **Shape**: {report['data_overview']['shape']}
- **Memory Usage**: {report['data_overview']['memory_usage_mb']:.2f} MB
- **Columns**: {len(report['data_overview']['columns'])}

## Missing Values
- **Total Missing**: {report['missingness']['total_missing']}
- **Missing Percentage**: {report['missingness']['missing_percentage']:.2f}%

## Duplicates
- **Exact Duplicates**: {report['duplicates']['exact_duplicates']}

## Time Series Analysis
- **Date Range**: {report['time_series']['date_range']['start']} to {report['time_series']['date_range']['end']}
- **Missing Dates**: {report['time_series']['missing_dates']['count']}

## Column Details
"""

        for col in report["data_overview"]["columns"]:
            md_content += f"- **{col}**: {report['data_overview']['dtypes'][col]}\n"

        with open(output_path, "w") as f:
            f.write(md_content)

        self.logger.info("Saved Markdown quality report", path=str(output_path))

    def _save_json_report(self, report: dict, output_path: Path) -> None:
        """Save JSON quality report.

        Args:
            report: Quality report data.
            output_path: Path to save JSON report.
        """
        import json

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info("Saved JSON quality report", path=str(output_path))

    def _generate_plots(self, data: pd.DataFrame, report: dict, output_dir: Path) -> None:
        """Generate quality visualization plots.

        Args:
            data: DataFrame to visualize.
            report: Quality report data.
            output_dir: Directory to save plots.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use("seaborn-v0_8")
            sns.set_palette("husl")

            # Missing values heatmap
            if data.isnull().sum().sum() > 0:
                plt.figure(figsize=(12, 8))
                sns.heatmap(data.isnull(), cbar=True, yticklabels=False)
                plt.title("Missing Values Heatmap")
                plt.tight_layout()
                plt.savefig(output_dir / "missing_values_heatmap.png", dpi=300, bbox_inches="tight")
                plt.close()

            # Sales distribution
            if "sales" in data.columns:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                data["sales"].hist(bins=50)
                plt.title("Sales Distribution")
                plt.xlabel("Sales")
                plt.ylabel("Frequency")

                plt.subplot(1, 2, 2)
                data["sales"].plot(kind="box")
                plt.title("Sales Box Plot")
                plt.ylabel("Sales")

                plt.tight_layout()
                plt.savefig(output_dir / "sales_distribution.png", dpi=300, bbox_inches="tight")
                plt.close()

            # Time series plot
            if "date" in data.columns and "sales" in data.columns:
                plt.figure(figsize=(15, 8))
                data_grouped = data.groupby("date")["sales"].sum()
                data_grouped.plot()
                plt.title("Total Sales Over Time")
                plt.xlabel("Date")
                plt.ylabel("Total Sales")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / "sales_timeseries.png", dpi=300, bbox_inches="tight")
                plt.close()

            self.logger.info("Generated quality plots", output_dir=str(output_dir))

        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available, skipping plots")
        except Exception as e:
            self.logger.error("Failed to generate plots", error=str(e))
