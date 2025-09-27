"""Streamlit dashboard for ForecastIt."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import structlog

from forecastit.config.settings import Settings
from forecastit.utils.logging import setup_logging

# Initialize settings and logging
settings = Settings()
setup_logging(settings)
logger = structlog.get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="ForecastIt Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_BASE_URL = f"http://{settings.api_host}:{settings.api_port}"


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load sample data for demonstration."""
    # This would typically load from your data source
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    sample_data = []
    for store_id in ["store_01", "store_02", "store_03"]:
        for item_id in ["item_001", "item_002", "item_003"]:
            for dt in dates:
                # Generate sample sales data
                base_sales = 100 + (hash(f"{store_id}_{item_id}") % 100)
                seasonal = 20 * np.sin(2 * np.pi * dt.timetuple().tm_yday / 365.25)
                noise = np.random.normal(0, 10)
                sales = max(0, base_sales + seasonal + noise)

                sample_data.append({
                    "date": dt,
                    "store_id": store_id,
                    "item_id": item_id,
                    "sales": sales,
                    "on_promo": np.random.choice([0, 1], p=[0.9, 0.1]),
                    "price": 10 + np.random.normal(0, 2),
                })

    return pd.DataFrame(sample_data)


def make_api_request(endpoint: str, data: dict) -> dict:
    """Make API request to ForecastIt backend."""
    try:
        import httpx

        with httpx.Client() as client:
            response = client.post(f"{API_BASE_URL}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"API request failed: {e}")
        return {}


def main():
    """Main Streamlit application."""
    st.title("📈 ForecastIt Dashboard")
    st.markdown("Intelligent demand & sales forecasting system")

    # Sidebar
    st.sidebar.header("Configuration")

    # Store and item selection
    stores = ["store_01", "store_02", "store_03"]
    items = ["item_001", "item_002", "item_003"]

    selected_store = st.sidebar.selectbox("Select Store", stores)
    selected_item = st.sidebar.selectbox("Select Item", items)

    # Date range selection
    st.sidebar.header("Forecast Period")
    forecast_start = st.sidebar.date_input(
        "Start Date",
        value=date.today() + timedelta(days=1),
        min_value=date.today(),
    )
    forecast_end = st.sidebar.date_input(
        "End Date",
        value=date.today() + timedelta(days=28),
        min_value=forecast_start,
    )

    # Scenario settings
    st.sidebar.header("Scenario Settings")
    promo_percentage = st.sidebar.slider(
        "Promotion Days (%)",
        min_value=0,
        max_value=100,
        value=10,
        step=5,
    )
    price_change = st.sidebar.slider(
        "Price Change (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
    )

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔮 Forecasts",
        "🎯 Scenarios",
        "📦 Inventory",
        "🔍 Explainability"
    ])

    with tab1:
        show_overview_tab(selected_store, selected_item)

    with tab2:
        show_forecast_tab(selected_store, selected_item, forecast_start, forecast_end)

    with tab3:
        show_scenario_tab(
            selected_store,
            selected_item,
            forecast_start,
            forecast_end,
            promo_percentage,
            price_change,
        )

    with tab4:
        show_inventory_tab(selected_store, selected_item)

    with tab5:
        show_explainability_tab(selected_store, selected_item)


def show_overview_tab(store_id: str, item_id: str):
    """Show overview tab."""
    st.header("📊 Business Overview")

    # Load sample data
    data = load_sample_data()
    filtered_data = data[
        (data["store_id"] == store_id) & (data["item_id"] == item_id)
    ]

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_sales = filtered_data["sales"].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")

    with col2:
        avg_sales = filtered_data["sales"].mean()
        st.metric("Avg Daily Sales", f"${avg_sales:.0f}")

    with col3:
        promo_days = filtered_data["on_promo"].sum()
        st.metric("Promo Days", f"{promo_days}")

    with col4:
        avg_price = filtered_data["price"].mean()
        st.metric("Avg Price", f"${avg_price:.2f}")

    # Sales trend chart
    st.subheader("Sales Trend")

    fig = px.line(
        filtered_data,
        x="date",
        y="sales",
        title=f"Sales Trend - {store_id} / {item_id}",
        labels={"sales": "Sales ($)", "date": "Date"},
    )

    # Add promotion markers
    promo_data = filtered_data[filtered_data["on_promo"] == 1]
    fig.add_scatter(
        x=promo_data["date"],
        y=promo_data["sales"],
        mode="markers",
        marker={"color": "red", "size": 8, "symbol": "diamond"},
        name="Promotions",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Price vs Sales scatter
    st.subheader("Price vs Sales Relationship")

    fig_scatter = px.scatter(
        filtered_data,
        x="price",
        y="sales",
        color="on_promo",
        title="Price vs Sales",
        labels={"price": "Price ($)", "sales": "Sales ($)"},
    )

    st.plotly_chart(fig_scatter, use_container_width=True)


def show_forecast_tab(store_id: str, item_id: str, start_date: date, end_date: date):
    """Show forecast tab."""
    st.header("🔮 Sales Forecasts")

    # Generate forecast button
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            # Make API request
            request_data = {
                "store_id": store_id,
                "item_id": item_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }

            forecasts = make_api_request("/predict", request_data)

            if forecasts:
                # Convert to DataFrame
                forecast_df = pd.DataFrame(forecasts)

                # Display forecast chart
                fig = go.Figure()

                # Add forecast line with confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_df["forecast_date"],
                    y=forecast_df["yhat"],
                    mode="lines",
                    name="Forecast",
                    line={"color": "blue", "width": 2},
                ))
                
                # Add confidence intervals if available
                if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_df["forecast_date"],
                        y=forecast_df["yhat_upper"],
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df["forecast_date"],
                        y=forecast_df["yhat_lower"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(0,100,80,0.2)",
                        name="Confidence Interval",
                        hoverinfo="skip",
                    ))


                fig.update_layout(
                    title=f"Sales Forecast - {store_id} / {item_id}",
                    xaxis_title="Date",
                    yaxis_title="Sales ($)",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display forecast table
                st.subheader("Forecast Details")
                st.dataframe(forecast_df)

                # Download button
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name=f"forecast_{store_id}_{item_id}_{start_date}_{end_date}.csv",
                    mime="text/csv",
                )
            else:
                st.error("Failed to generate forecast")


def show_scenario_tab(
    store_id: str,
    item_id: str,
    start_date: date,
    end_date: date,
    promo_percentage: int,
    price_change: int,
):
    """Show scenario analysis tab."""
    st.header("🎯 Scenario Analysis")

    st.subheader("Scenario Settings")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Promotion Days", f"{promo_percentage}%")

    with col2:
        st.metric("Price Change", f"{price_change:+}%")

    # Generate scenario forecast
    if st.button("Run Scenario Analysis", type="primary"):
        with st.spinner("Running scenario analysis..."):
            # Create promo plan
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            promo_plan = []
            price_plan = []

            for _date in date_range:
                # Randomly assign promotions based on percentage
                is_promo = np.random.random() < (promo_percentage / 100)
                promo_plan.append(1 if is_promo else 0)

                # Apply price change
                base_price = 10.0  # Would get from historical data
                price_plan.append(base_price * (1 + price_change / 100))

            # Make API request
            request_data = {
                "store_id": store_id,
                "item_id": item_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "promo_plan": promo_plan,
                "price_plan": price_plan,
            }

            scenario_forecasts = make_api_request("/predict", request_data)

            if scenario_forecasts:
                # Compare with baseline
                baseline_data = {
                    "store_id": store_id,
                    "item_id": item_id,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                }

                baseline_forecasts = make_api_request("/predict", baseline_data)

                if baseline_forecasts:
                    # Create comparison chart
                    scenario_df = pd.DataFrame(scenario_forecasts)
                    baseline_df = pd.DataFrame(baseline_forecasts)

                    fig = go.Figure()

                    # Baseline forecast
                    fig.add_trace(go.Scatter(
                        x=baseline_df["forecast_date"],
                        y=baseline_df["yhat"],
                        mode="lines",
                        name="Baseline Forecast",
                        line={"color": "blue", "width": 2},
                    ))

                    # Scenario forecast
                    fig.add_trace(go.Scatter(
                        x=scenario_df["forecast_date"],
                        y=scenario_df["yhat"],
                        mode="lines",
                        name="Scenario Forecast",
                        line={"color": "red", "width": 2},
                    ))

                    fig.update_layout(
                        title="Baseline vs Scenario Forecast",
                        xaxis_title="Date",
                        yaxis_title="Sales ($)",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate impact
                    baseline_total = baseline_df["yhat"].sum()
                    scenario_total = scenario_df["yhat"].sum()
                    impact = scenario_total - baseline_total
                    impact_pct = (impact / baseline_total) * 100

                    st.subheader("Scenario Impact")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Baseline Total", f"${baseline_total:,.0f}")

                    with col2:
                        st.metric("Scenario Total", f"${scenario_total:,.0f}")

                    with col3:
                        st.metric("Impact", f"${impact:+,.0f} ({impact_pct:+.1f}%)")


def show_inventory_tab(store_id: str, item_id: str):
    """Show inventory optimization tab."""
    st.header("📦 Inventory Optimization")

    # Inventory parameters
    st.subheader("Inventory Parameters")

    col1, col2 = st.columns(2)

    with col1:
        service_level = st.slider(
            "Service Level",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
        )

        lead_time = st.number_input(
            "Lead Time (days)",
            min_value=1,
            max_value=30,
            value=7,
        )

    with col2:
        holding_cost = st.number_input(
            "Holding Cost ($/unit/day)",
            min_value=0.01,
            max_value=1.00,
            value=0.10,
            step=0.01,
        )

        stockout_cost = st.number_input(
            "Stockout Cost ($/unit)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
        )

    # Calculate inventory suggestions
    if st.button("Calculate Inventory Suggestions", type="primary"):
        with st.spinner("Calculating inventory suggestions..."):
            request_data = {
                "store_id": store_id,
                "item_id": item_id,
                "service_level": service_level,
                "lead_time": lead_time,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
            }

            inventory_suggestions = make_api_request("/inventory/suggest", request_data)

            if inventory_suggestions:
                st.subheader("Inventory Recommendations")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Reorder Point",
                        f"{inventory_suggestions['reorder_point']:.0f}",
                    )

                with col2:
                    st.metric(
                        "Safety Stock",
                        f"{inventory_suggestions['safety_stock']:.0f}",
                    )

                with col3:
                    st.metric(
                        "Expected Demand",
                        f"{inventory_suggestions['expected_demand']:.0f}",
                    )

                with col4:
                    st.metric(
                        "Stockout Probability",
                        f"{inventory_suggestions['stockout_probability']:.1%}",
                    )

                # Inventory visualization
                fig = go.Figure()

                # Add inventory levels
                fig.add_hline(
                    y=inventory_suggestions["reorder_point"],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Reorder Point",
                )

                fig.add_hline(
                    y=inventory_suggestions["safety_stock"],
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="Safety Stock",
                )

                fig.update_layout(
                    title="Inventory Levels",
                    xaxis_title="Time",
                    yaxis_title="Inventory Level",
                    showlegend=True,
                )

                st.plotly_chart(fig, use_container_width=True)


def show_explainability_tab(store_id: str, item_id: str):
    """Show explainability tab."""
    st.header("🔍 Model Explainability")

    # Get feature importance
    if st.button("Generate Explanation", type="primary"):
        with st.spinner("Generating explanation..."):
            explanation = make_api_request(f"/explain/{store_id}/{item_id}", {})

            if explanation and "top_features" in explanation:
                st.subheader("Top Contributing Features")

                # Feature importance chart
                top_features = explanation["top_features"][:10]
                features_df = pd.DataFrame(top_features)

                fig = px.bar(
                    features_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top Contributing Features",
                    color="importance",
                    color_continuous_scale="viridis",
                )
                fig.update_layout(height=500)

                st.plotly_chart(fig, use_container_width=True)

                # SHAP plot if available
                if "plot_path" in explanation and explanation["plot_path"]:
                    st.subheader("SHAP Feature Importance Plot")
                    try:
                        st.image(explanation["plot_path"], caption="SHAP Beeswarm Plot")
                    except Exception as e:
                        st.warning(f"Could not display SHAP plot: {e}")

                # Feature importance table
                st.subheader("Detailed Feature Importance")
                st.dataframe(features_df)

                # Model information
                st.subheader("Model Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Explanation Type: {explanation.get('explanation_type', 'N/A')}")
                with col2:
                    st.info(f"Model: {explanation.get('model_name', 'N/A')}")
            else:
                st.error("Failed to generate explanation")


def show_backtest_tab():
    """Show backtesting results tab."""
    st.header("📊 Model Performance & Backtesting")

    # This would typically load from MLflow runs
    st.subheader("Model Leaderboard")

    # Mock backtest results
    backtest_results = pd.DataFrame({
        "Model": ["LightGBM", "XGBoost", "Prophet", "SARIMAX", "Naive"],
        "sMAPE": [12.5, 13.2, 14.8, 15.3, 18.7],
        "RMSE": [45.2, 47.1, 52.3, 55.1, 68.4],
        "MAPE": [11.8, 12.4, 13.9, 14.2, 17.6],
        "Folds": [5, 5, 5, 5, 5],
    })

    # Sort by sMAPE (primary metric)
    backtest_results = backtest_results.sort_values("sMAPE")

    # Display leaderboard
    st.dataframe(backtest_results, use_container_width=True)

    # Performance comparison chart
    st.subheader("Model Performance Comparison")

    fig = go.Figure()

    for metric in ["sMAPE", "RMSE", "MAPE"]:
        fig.add_trace(go.Bar(
            name=metric,
            x=backtest_results["Model"],
            y=backtest_results[metric],
        ))

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Metric Value",
        barmode="group",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Best model info
    best_model = backtest_results.iloc[0]
    st.success(f"🏆 Best Model: {best_model['Model']} (sMAPE: {best_model['sMAPE']:.1f}%)")


if __name__ == "__main__":
    main()
