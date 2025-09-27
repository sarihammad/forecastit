"""Inventory optimization calculations."""

from typing import Any

import numpy as np
import structlog
from scipy import stats

from forecastit.config.settings import Settings

logger = structlog.get_logger(__name__)


class InventoryOptimizer:
    """Inventory optimization calculations."""

    def __init__(self, settings: Settings):
        """Initialize inventory optimizer.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.logger = logger.bind(component="inventory_optimizer")

    def calculate_inventory_metrics(
        self,
        forecasts: list[dict[str, Any]],
        service_level: float = 0.95,
        lead_time: int = 7,
        holding_cost: float = 0.1,
        stockout_cost: float = 10.0,
    ) -> dict[str, float]:
        """Calculate inventory optimization metrics.

        Args:
            forecasts: List of forecast dictionaries.
            service_level: Target service level.
            lead_time: Lead time in days.
            holding_cost: Holding cost per unit per day.
            stockout_cost: Stockout cost per unit.

        Returns:
            Dictionary with inventory metrics.
        """
        try:
            self.logger.info(
                "Calculating inventory metrics",
                forecast_count=len(forecasts),
                service_level=service_level,
                lead_time=lead_time,
            )

            # Extract forecast values
            point_forecasts = [f["yhat"] for f in forecasts[:lead_time]]
            lower_bounds = [f["yhat_lower"] for f in forecasts[:lead_time]]
            upper_bounds = [f["yhat_upper"] for f in forecasts[:lead_time]]

            # Calculate demand statistics
            expected_demand = sum(point_forecasts)
            demand_std = self._calculate_demand_std(lower_bounds, upper_bounds)

            # Calculate safety stock
            safety_stock = self._calculate_safety_stock(
                demand_std=demand_std,
                lead_time=lead_time,
                service_level=service_level,
            )

            # Calculate reorder point
            reorder_point = expected_demand + safety_stock

            # Calculate stockout probability
            stockout_probability = 1 - service_level

            # Calculate economic order quantity (EOQ)
            eoq = self._calculate_eoq(
                expected_demand=expected_demand,
                holding_cost=holding_cost,
                stockout_cost=stockout_cost,
            )

            metrics = {
                "expected_demand": expected_demand,
                "demand_std": demand_std,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "stockout_probability": stockout_probability,
                "service_level": service_level,
                "economic_order_quantity": eoq,
                "lead_time": lead_time,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
            }

            self.logger.info(
                "Calculated inventory metrics",
                expected_demand=expected_demand,
                safety_stock=safety_stock,
                reorder_point=reorder_point,
            )

            return metrics

        except Exception as e:
            self.logger.error("Failed to calculate inventory metrics", error=str(e))
            raise

    def _calculate_demand_std(
        self,
        lower_bounds: list[float],
        upper_bounds: list[float],
    ) -> float:
        """Calculate demand standard deviation from prediction intervals.

        Args:
            lower_bounds: Lower prediction bounds.
            upper_bounds: Upper prediction bounds.

        Returns:
            Estimated demand standard deviation.
        """
        # Estimate standard deviation from prediction intervals
        # Assuming 95% confidence intervals (approximately 1.96 * std)
        interval_widths = [upper - lower for upper, lower in zip(upper_bounds, lower_bounds)]
        avg_interval_width = np.mean(interval_widths)

        # Convert to standard deviation (assuming normal distribution)
        # For 95% confidence interval: width = 2 * 1.96 * std
        demand_std = avg_interval_width / (2 * 1.96)

        return max(demand_std, 0.1)  # Minimum standard deviation

    def _calculate_safety_stock(
        self,
        demand_std: float,
        lead_time: int,
        service_level: float,
    ) -> float:
        """Calculate safety stock using the square root law.

        Args:
            demand_std: Demand standard deviation.
            lead_time: Lead time in days.
            service_level: Target service level.

        Returns:
            Safety stock quantity.
        """
        # Z-score for service level
        z_score = stats.norm.ppf(service_level)

        # Safety stock = z * std * sqrt(lead_time)
        safety_stock = z_score * demand_std * np.sqrt(lead_time)

        return max(safety_stock, 0)

    def _calculate_eoq(
        self,
        expected_demand: float,
        holding_cost: float,
        stockout_cost: float,
    ) -> float:
        """Calculate Economic Order Quantity.

        Args:
            expected_demand: Expected annual demand.
            holding_cost: Holding cost per unit per year.
            stockout_cost: Stockout cost per unit.

        Returns:
            Economic order quantity.
        """
        # Annualize demand (assuming daily forecasts)
        annual_demand = expected_demand * 365

        # EOQ formula (simplified)
        # EOQ = sqrt(2 * D * S / H)
        # Where D = demand, S = ordering cost (assumed), H = holding cost

        ordering_cost = 10.0  # Assumed ordering cost

        eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost) if holding_cost > 0 else 0

        return max(eoq, 1)  # Minimum order quantity

    def calculate_optimal_reorder_point(
        self,
        demand_mean: float,
        demand_std: float,
        lead_time: int,
        service_level: float,
    ) -> float:
        """Calculate optimal reorder point.

        Args:
            demand_mean: Mean demand during lead time.
            demand_std: Standard deviation of demand during lead time.
            lead_time: Lead time in days.
            service_level: Target service level.

        Returns:
            Optimal reorder point.
        """
        # Expected demand during lead time
        expected_demand_lt = demand_mean * lead_time

        # Safety stock
        safety_stock = self._calculate_safety_stock(
            demand_std=demand_std,
            lead_time=lead_time,
            service_level=service_level,
        )

        # Reorder point = expected demand + safety stock
        reorder_point = expected_demand_lt + safety_stock

        return max(reorder_point, 0)

    def calculate_stockout_probability(
        self,
        current_stock: float,
        demand_mean: float,
        demand_std: float,
        lead_time: int,
    ) -> float:
        """Calculate probability of stockout.

        Args:
            current_stock: Current inventory level.
            demand_mean: Mean demand during lead time.
            demand_std: Standard deviation of demand during lead time.
            lead_time: Lead time in days.

        Returns:
            Probability of stockout.
        """
        # Expected demand during lead time
        expected_demand_lt = demand_mean * lead_time

        # Demand standard deviation during lead time
        demand_std_lt = demand_std * np.sqrt(lead_time)

        if demand_std_lt > 0:
            # Z-score
            z_score = (current_stock - expected_demand_lt) / demand_std_lt

            # Stockout probability = P(demand > current_stock)
            stockout_probability = 1 - stats.norm.cdf(z_score)
        else:
            stockout_probability = 1 if current_stock < expected_demand_lt else 0

        return max(0, min(1, stockout_probability))

    def calculate_inventory_turnover(
        self,
        annual_sales: float,
        average_inventory: float,
    ) -> float:
        """Calculate inventory turnover ratio.

        Args:
            annual_sales: Annual sales volume.
            average_inventory: Average inventory level.

        Returns:
            Inventory turnover ratio.
        """
        turnover = annual_sales / average_inventory if average_inventory > 0 else 0

        return turnover

    def calculate_carrying_cost(
        self,
        average_inventory: float,
        holding_cost_rate: float = 0.25,
    ) -> float:
        """Calculate annual carrying cost.

        Args:
            average_inventory: Average inventory level.
            holding_cost_rate: Annual holding cost rate.

        Returns:
            Annual carrying cost.
        """
        return average_inventory * holding_cost_rate

    def optimize_order_quantity(
        self,
        demand_forecast: list[float],
        holding_cost: float,
        ordering_cost: float,
        stockout_cost: float,
    ) -> dict[str, float]:
        """Optimize order quantity using various methods.

        Args:
            demand_forecast: Demand forecast for the period.
            holding_cost: Holding cost per unit.
            ordering_cost: Ordering cost per order.
            stockout_cost: Stockout cost per unit.

        Returns:
            Dictionary with optimization results.
        """
        expected_demand = np.mean(demand_forecast)
        demand_std = np.std(demand_forecast)

        # EOQ
        eoq = self._calculate_eoq(expected_demand, holding_cost, stockout_cost)

        # Newsvendor optimal quantity
        critical_ratio = stockout_cost / (stockout_cost + holding_cost)
        newsvendor_quantity = stats.norm.ppf(critical_ratio) * demand_std + expected_demand

        # Service level optimization
        service_level_95 = self.calculate_optimal_reorder_point(
            demand_mean=expected_demand,
            demand_std=demand_std,
            lead_time=7,  # Assume 7-day lead time
            service_level=0.95,
        )

        return {
            "eoq": eoq,
            "newsvendor_quantity": max(newsvendor_quantity, 0),
            "service_level_95_reorder": service_level_95,
            "expected_demand": expected_demand,
            "demand_std": demand_std,
            "critical_ratio": critical_ratio,
        }
