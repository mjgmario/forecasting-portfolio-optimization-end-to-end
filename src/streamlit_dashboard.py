"""
Streamlit Dashboard - Interactive Portfolio Forecasting

This dashboard provides both visualization of stored results AND interactive
local testing capabilities for forecasters and allocators.

Features:
1. Read-only mode: Visualize results from database
2. Interactive mode: Run forecasters and compare allocators locally

Environment Variables:
    DATABASE_URL: Database connection string (Neon PostgreSQL or SQLite)
"""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import get_config
from src.database import get_db

# Load configuration
config = get_config()
PORTFOLIO_TICKERS = config.portfolio.tickers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Portfolio Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Data Loading Functions (Read-Only Mode)
# =============================================================================


@st.cache_data(ttl=300)
def load_latest_forecasts(n=100):
    """Load latest forecasts from database."""
    try:
        db = get_db()
        df = db.get_latest_forecasts(n=n)
        if not df.empty:
            for col in ["created_at", "as_of_date", "prediction_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        return df
    except Exception as e:
        logger.error(f"Failed to load forecasts: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_latest_portfolios(n=50):
    """Load latest portfolio allocations from database."""
    try:
        db = get_db()
        df = db.get_latest_portfolios(n=n)
        if not df.empty:
            for col in ["created_at", "as_of_date", "rebalance_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        return df
    except Exception as e:
        logger.error(f"Failed to load portfolios: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_backtest_results():
    """Load all backtesting results from database."""
    try:
        db = get_db()
        df = db.get_backtest_results()
        if not df.empty:
            for col in ["created_at", "start_date", "end_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        return df
    except Exception as e:
        logger.error(f"Failed to load backtest results: {e}")
        return pd.DataFrame()


# =============================================================================
# Interactive Mode Functions
# =============================================================================


@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def fetch_market_data(tickers: list, period: str = "2y"):
    """Fetch market data for selected tickers."""
    from src.extractor import DataExtractor

    extractor = DataExtractor()
    return extractor.fetch_prices(tickers, period=period)


def get_available_forecasters():
    """Get list of available forecasters."""
    from src.forecasters import ForecasterRegistry

    return ForecasterRegistry.list_all()


def get_available_allocators():
    """Get list of available allocators."""
    from src.allocators import AllocatorRegistry

    return AllocatorRegistry.list_all()


def run_forecaster_comparison(
    prices_df: pd.DataFrame,
    forecasters: list,
    horizon: int = 21,
    combined_training: bool = False,
    progress_callback=None,
):
    """
    Run and compare multiple forecasters.

    Args:
        prices_df: DataFrame with price data
        forecasters: List of forecaster names to compare
        horizon: Forecast horizon
        combined_training: If True, train one model on all tickers (ML only)
        progress_callback: Callback for progress updates
    """
    from src.evaluation import BacktestEngine
    from src.forecasters import ForecasterRegistry

    results = []
    total = len(forecasters)

    for idx, forecaster_name in enumerate(forecasters):
        if progress_callback:
            progress_callback((idx + 1) / total, f"Testing {forecaster_name}...")

        try:
            forecaster = ForecasterRegistry.create(forecaster_name, config={"horizon": horizon})

            # Run backtest
            engine = BacktestEngine(
                forecaster=forecaster,
                initial_train_size=252,
                step_size=21,
                horizon=horizon,
            )

            backtest_results = engine.run(prices_df)

            # Aggregate metrics
            metrics = engine.aggregate_metrics(backtest_results)

            results.append(
                {
                    "forecaster": forecaster_name,
                    "family": getattr(forecaster, "family", "unknown"),
                    "mae": metrics.get("mae", np.nan),
                    "rmse": metrics.get("rmse", np.nan),
                    "mase": metrics.get("mase", np.nan),
                    "directional_accuracy": metrics.get("directional_accuracy", np.nan),
                }
            )

        except Exception as e:
            logger.error(f"Error running {forecaster_name}: {e}")
            results.append(
                {
                    "forecaster": forecaster_name,
                    "family": "error",
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mase": np.nan,
                    "directional_accuracy": np.nan,
                    "error": str(e),
                }
            )

    return pd.DataFrame(results)


def run_allocator_comparison(
    prices_df: pd.DataFrame,
    predicted_returns: dict,
    allocators: list,
):
    """
    Compare multiple allocators given predicted returns.

    Args:
        prices_df: Historical price data
        predicted_returns: Dict of {ticker: predicted_return}
        allocators: List of allocator names to compare
    """
    from src.allocators import AllocatorRegistry

    results = []

    # Calculate historical returns for allocators
    historical_returns = prices_df.pct_change().dropna()

    for allocator_name in allocators:
        try:
            allocator = AllocatorRegistry.create(allocator_name)
            weights = allocator.allocate(predicted_returns, historical_returns)

            # Calculate expected return
            expected_return = sum(
                weights.get(t, 0) * predicted_returns.get(t, 0) for t in predicted_returns.keys()
            )

            # Calculate portfolio volatility
            tickers = list(weights.keys())
            w = np.array([weights.get(t, 0) for t in tickers])
            cov_matrix = historical_returns[tickers].cov().values
            portfolio_vol = np.sqrt(w @ cov_matrix @ w) * np.sqrt(252)

            results.append(
                {
                    "allocator": allocator_name,
                    "expected_return": expected_return,
                    "portfolio_volatility": portfolio_vol,
                    "sharpe_ratio": expected_return / portfolio_vol if portfolio_vol > 0 else 0,
                    "weights": weights,
                    "max_weight": max(weights.values()),
                    "n_positions": sum(1 for w in weights.values() if w > 0.01),
                }
            )

        except Exception as e:
            logger.error(f"Error running {allocator_name}: {e}")
            results.append(
                {
                    "allocator": allocator_name,
                    "expected_return": np.nan,
                    "portfolio_volatility": np.nan,
                    "sharpe_ratio": np.nan,
                    "weights": {},
                    "error": str(e),
                }
            )

    return pd.DataFrame(results)


# =============================================================================
# UI Rendering Functions
# =============================================================================


def render_header():
    """Render dashboard header."""
    st.markdown(
        '<div class="main-header">📈 Portfolio Forecasting Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Multi-Model Forecasting & Portfolio Optimization System**")
    st.markdown("---")


def render_overview():
    """Render overview section with key metrics."""
    st.header("📊 Overview")

    forecasts_df = load_latest_forecasts(n=100)
    portfolios_df = load_latest_portfolios(n=20)

    if forecasts_df.empty and portfolios_df.empty:
        st.warning("No data available. Run forecasts using GitHub Actions or use Interactive Mode.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        n_forecasts = len(forecasts_df)
        st.metric("Total Forecasts", f"{n_forecasts:,}")

    with col2:
        n_tickers = forecasts_df["ticker"].nunique() if not forecasts_df.empty else 0
        st.metric("Assets Tracked", n_tickers)

    with col3:
        n_models = forecasts_df["forecaster"].nunique() if not forecasts_df.empty else 0
        st.metric("Models Used", n_models)

    with col4:
        if not forecasts_df.empty:
            last_update = forecasts_df["created_at"].max()
            st.metric("Last Update", last_update.strftime("%Y-%m-%d"))
        else:
            st.metric("Last Update", "N/A")

    st.markdown("---")


def render_latest_forecasts():
    """Render latest forecasts section."""
    st.header("🔮 Latest Forecasts")

    forecasts_df = load_latest_forecasts(n=50)

    if forecasts_df.empty:
        st.info("No forecast data available yet. Use Interactive Mode to run forecasts.")
        return

    col1, col2 = st.columns(2)

    with col1:
        unique_dates = sorted(forecasts_df["as_of_date"].dt.date.unique(), reverse=True)
        selected_date = st.selectbox("Select Date:", unique_dates)

    with col2:
        unique_forecasters = sorted(forecasts_df["forecaster"].unique())
        selected_forecaster = st.selectbox("Select Forecaster:", ["All"] + unique_forecasters)

    filtered_df = forecasts_df[forecasts_df["as_of_date"].dt.date == selected_date]

    if selected_forecaster != "All":
        filtered_df = filtered_df[filtered_df["forecaster"] == selected_forecaster]

    if filtered_df.empty:
        st.warning("No data for selected filters.")
        return

    display_cols = ["ticker", "forecaster", "predicted_price", "predicted_return"]
    display_df = filtered_df[display_cols].copy()
    display_df["predicted_return"] = display_df["predicted_return"].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Predicted Returns")

    fig = px.bar(
        filtered_df,
        x="ticker",
        y="predicted_return",
        color="predicted_return",
        color_continuous_scale="RdYlGn",
        labels={"predicted_return": "Return", "ticker": "Ticker"},
        title=f"Predicted Returns - {selected_date}",
    )

    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Predicted Return (%)",
        yaxis_tickformat=".1%",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_portfolio_allocations():
    """Render portfolio allocations section."""
    st.header("💼 Portfolio Allocations")

    portfolios_df = load_latest_portfolios(n=20)

    if portfolios_df.empty:
        st.info("No portfolio data available yet.")
        return

    unique_dates = sorted(portfolios_df["as_of_date"].dt.date.unique(), reverse=True)
    selected_date = st.selectbox("Select Portfolio Date:", unique_dates, key="portfolio_date")

    portfolio = portfolios_df[portfolios_df["as_of_date"].dt.date == selected_date].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Forecaster", portfolio["forecaster"])

    with col2:
        st.metric("Allocator", portfolio["allocator"])

    with col3:
        st.metric("Date", str(selected_date))

    weights = portfolio["weights"]

    if isinstance(weights, str):
        weights = json.loads(weights)

    weights_df = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
    weights_df = weights_df.sort_values("Weight", ascending=False)

    st.subheader("Allocation Weights")
    st.dataframe(
        weights_df.style.format({"Weight": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.pie(
        weights_df,
        values="Weight",
        names="Ticker",
        title="Portfolio Allocation",
        hole=0.4,
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")

    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison():
    """Render model comparison section."""
    st.header("🏆 Model Comparison")

    backtests_df = load_backtest_results()

    if backtests_df.empty:
        st.info("No backtest results available yet. Use Interactive Mode to run comparisons.")
        return

    latest_run = backtests_df["created_at"].max()
    latest_backtests = backtests_df[backtests_df["created_at"] == latest_run]

    st.subheader(f"Latest Backtest Run: {latest_run.strftime('%Y-%m-%d %H:%M')}")

    metrics = ["mae", "rmse", "mase", "directional_accuracy"]

    display_df = latest_backtests[["forecaster"] + metrics].copy()

    display_df["directional_accuracy"] = display_df["directional_accuracy"].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    )

    for metric in ["mae", "rmse", "mase"]:
        display_df[metric] = display_df[metric].apply(
            lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
        )

    display_df = display_df.sort_values("mase")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("MASE Comparison (Lower is Better)")

    fig = px.bar(
        latest_backtests.sort_values("mase"),
        x="forecaster",
        y="mase",
        color="mase",
        color_continuous_scale="RdYlGn_r",
        labels={"mase": "MASE", "forecaster": "Forecaster"},
    )

    fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Naive Baseline")

    st.plotly_chart(fig, use_container_width=True)


def render_interactive_mode():
    """Render interactive testing mode."""
    st.header("🧪 Interactive Testing Mode")

    st.markdown(
        """
        Run forecasters and allocators locally to test different strategies.
        This mode fetches live market data and runs models in real-time.
        """
    )

    # Configuration
    st.subheader("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Ticker selection
        default_tickers = ["AAPL", "MSFT", "NVDA", "AMD", "GOOG"]
        selected_tickers = st.multiselect(
            "Select Tickers:",
            options=PORTFOLIO_TICKERS,
            default=default_tickers,
            help="Select assets to include in the analysis",
        )

        # Data period
        data_period = st.selectbox(
            "Data Period:",
            options=["1y", "2y", "5y"],
            index=1,
            help="Historical data period for training",
        )

    with col2:
        # Forecast horizon
        horizon = st.selectbox(
            "Forecast Horizon (days):",
            options=[1, 5, 21, 63, 252],
            index=2,
            format_func=lambda x: {
                1: "1 day",
                5: "1 week",
                21: "1 month",
                63: "1 quarter",
                252: "1 year",
            }.get(x, f"{x} days"),
        )

        # Training mode
        training_mode = st.radio(
            "ML Training Mode:",
            options=["separate", "combined"],
            format_func=lambda x: {
                "separate": "Separate (one model per ticker)",
                "combined": "Combined (one model for all tickers)",
            }.get(x),
            help="Combined mode pools data from all tickers to train a single model",
        )

    st.markdown("---")

    # Forecaster selection
    st.subheader("Select Forecasters")

    available_forecasters = get_available_forecasters()

    # Filter out skipped forecasters from config
    skip_forecasters = config.forecasters.skip
    available_forecasters = [f for f in available_forecasters if f not in skip_forecasters]

    # Group forecasters by family
    forecaster_families = {
        "baselines": ["naive", "seasonal_naive", "drift"],
        "statistical": ["ets", "theta", "arima"],
        "ml": [
            "ridge",
            "lasso",
            "elasticnet",
            "random_forest",
            "hist_gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
        ],
        "volatility": ["garch", "egarch", "gjr_garch"],
    }

    # Filter families by available forecasters
    for family in forecaster_families:
        forecaster_families[family] = [
            f for f in forecaster_families[family] if f in available_forecasters
        ]

    # Show default from config
    st.info(
        f"**Config default:** {config.forecasters.default} | **Training mode:** {config.forecasters.training_mode}"
    )

    col1, col2 = st.columns(2)

    with col1:
        quick_select = st.radio(
            "Quick Select:",
            options=[
                "Config Default",
                "Custom",
                "All",
                "Fast (Baselines + ML)",
                "Statistical Only",
            ],
        )

    selected_forecasters = []

    if quick_select == "Config Default":
        selected_forecasters = [config.forecasters.default]
    elif quick_select == "All":
        selected_forecasters = available_forecasters
    elif quick_select == "Fast (Baselines + ML)":
        selected_forecasters = forecaster_families["baselines"] + forecaster_families["ml"]
    elif quick_select == "Statistical Only":
        selected_forecasters = forecaster_families["statistical"]
    else:
        with col2:
            selected_forecasters = st.multiselect(
                "Select Forecasters:",
                options=available_forecasters,
                default=[config.forecasters.default],
            )

    st.markdown("---")

    # Allocator selection
    st.subheader("Select Allocators")

    available_allocators = get_available_allocators()

    # Show default from config
    st.info(f"**Config default:** {config.allocators.default}")

    allocator_select = st.radio(
        "Allocator Selection:",
        options=["Config Default", "All", "Simple", "Custom"],
    )

    if allocator_select == "Config Default":
        selected_allocators = [config.allocators.default]
    elif allocator_select == "All":
        selected_allocators = available_allocators
    elif allocator_select == "Simple":
        selected_allocators = ["equal_weight", "inverse_volatility", "risk_parity"]
    else:
        selected_allocators = st.multiselect(
            "Select Allocators:",
            options=available_allocators,
            default=[config.allocators.default],
        )

    st.markdown("---")

    # Run button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not selected_tickers:
            st.error("Please select at least one ticker.")
            return

        if not selected_forecasters:
            st.error("Please select at least one forecaster.")
            return

        # Fetch data
        with st.spinner("Fetching market data..."):
            prices_df = fetch_market_data(selected_tickers, period=data_period)

        if prices_df.empty:
            st.error("Failed to fetch market data. Please try again.")
            return

        st.success(f"Loaded {len(prices_df)} days of data for {len(selected_tickers)} tickers.")

        # Run forecaster comparison
        st.subheader("📊 Forecaster Comparison")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress, text):
            progress_bar.progress(progress)
            status_text.text(text)

        forecaster_results = run_forecaster_comparison(
            prices_df,
            selected_forecasters,
            horizon=horizon,
            combined_training=(training_mode == "combined"),
            progress_callback=update_progress,
        )

        progress_bar.empty()
        status_text.empty()

        # Display forecaster results
        valid_results = forecaster_results.dropna(subset=["mase"])

        if valid_results.empty:
            st.error("No forecasters completed successfully.")
            return

        # Sort by MASE
        valid_results = valid_results.sort_values("mase")

        st.dataframe(
            valid_results[
                ["forecaster", "family", "mae", "mase", "directional_accuracy"]
            ].style.format(
                {
                    "mae": "{:.6f}",
                    "mase": "{:.4f}",
                    "directional_accuracy": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Visualize
        fig = px.bar(
            valid_results,
            x="forecaster",
            y="mase",
            color="family",
            title="MASE by Forecaster (Lower is Better)",
            labels={"mase": "MASE", "forecaster": "Forecaster"},
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Naive Baseline")
        st.plotly_chart(fig, use_container_width=True)

        # Best forecaster
        best_forecaster = valid_results.iloc[0]["forecaster"]
        best_mase = valid_results.iloc[0]["mase"]

        st.markdown(
            f"""
            <div class="success-box">
            <strong>Best Forecaster:</strong> {best_forecaster} (MASE: {best_mase:.4f})
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Run allocator comparison with best forecaster
        st.subheader("💼 Allocator Comparison")

        with st.spinner("Running allocator comparison..."):
            # Generate predictions using best forecaster
            from src.forecasters import ForecasterRegistry

            best_model = ForecasterRegistry.create(best_forecaster, config={"horizon": horizon})

            predicted_returns = {}
            for ticker in selected_tickers:
                try:
                    ticker_prices = prices_df[ticker].dropna()
                    best_model.fit(ticker_prices)
                    forecast = best_model.predict(horizon=horizon)
                    predicted_returns[ticker] = (forecast.iloc[-1] / ticker_prices.iloc[-1]) - 1
                except Exception as e:
                    logger.warning(f"Failed to predict {ticker}: {e}")
                    predicted_returns[ticker] = 0.0

            allocator_results = run_allocator_comparison(
                prices_df, predicted_returns, selected_allocators
            )

        # Display allocator results
        valid_allocators = allocator_results.dropna(subset=["expected_return"])

        if valid_allocators.empty:
            st.error("No allocators completed successfully.")
            return

        valid_allocators = valid_allocators.sort_values("sharpe_ratio", ascending=False)

        st.dataframe(
            valid_allocators[
                [
                    "allocator",
                    "expected_return",
                    "portfolio_volatility",
                    "sharpe_ratio",
                    "n_positions",
                ]
            ].style.format(
                {
                    "expected_return": "{:.2%}",
                    "portfolio_volatility": "{:.2%}",
                    "sharpe_ratio": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Visualize allocator results
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=valid_allocators["portfolio_volatility"],
                y=valid_allocators["expected_return"],
                mode="markers+text",
                text=valid_allocators["allocator"],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=valid_allocators["sharpe_ratio"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Sharpe"),
                ),
            )
        )

        fig.update_layout(
            title="Risk-Return Profile by Allocator",
            xaxis_title="Portfolio Volatility (Annualized)",
            yaxis_title="Expected Return",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".2%",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Best allocator
        best_allocator = valid_allocators.iloc[0]["allocator"]
        best_sharpe = valid_allocators.iloc[0]["sharpe_ratio"]

        st.markdown(
            f"""
            <div class="success-box">
            <strong>Best Allocator:</strong> {best_allocator} (Sharpe: {best_sharpe:.3f})
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show best portfolio weights
        st.subheader("🎯 Recommended Portfolio")

        best_weights = valid_allocators.iloc[0]["weights"]

        if best_weights:
            weights_df = pd.DataFrame(
                [(k, v) for k, v in best_weights.items() if v > 0.01],
                columns=["Ticker", "Weight"],
            ).sort_values("Weight", ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(
                    weights_df.style.format({"Weight": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                fig = px.pie(
                    weights_df,
                    values="Weight",
                    names="Ticker",
                    title=f"{best_allocator} Portfolio",
                    hole=0.4,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)


def render_sidebar():
    """Render sidebar with info and controls."""
    with st.sidebar:
        st.title("ℹ️ About")

        st.markdown(
            """
            This dashboard visualizes results from a multi-model
            forecasting and portfolio optimization system.

            **Key Features:**
            - 19 forecasting models
            - 14 portfolio allocation methods
            - Walk-forward backtesting
            - Interactive local testing
            - Combined training mode (ML)
            """
        )

        st.markdown("---")

        st.subheader("📚 Available Models")

        with st.expander("Forecasters (19)"):
            st.markdown(
                """
                **Baselines:**
                - Naive, Seasonal Naive, Drift

                **Statistical:**
                - ETS, Theta, ARIMA

                **Volatility:**
                - GARCH, EGARCH, GJR-GARCH

                **Machine Learning:**
                - Ridge, Lasso, ElasticNet
                - Random Forest, XGBoost
                - HistGradientBoosting
                - LightGBM*, CatBoost*

                **Deep Learning:**
                - NBEATS*, NHITS*

                *Optional dependencies
                """
            )

        with st.expander("Allocators (14)"):
            st.markdown(
                """
                **Simple:**
                - Equal Weight (1/N)
                - Inverse Volatility

                **Risk-Based:**
                - Risk Parity
                - Minimum Variance
                - Maximum Diversification

                **Mean-Variance:**
                - Markowitz (Classic)
                - Robust Markowitz
                - Maximum Sharpe

                **Advanced:**
                - Black-Litterman
                - Hierarchical Risk Parity
                - CVaR Optimization
                - Kelly Criterion
                - Mean-CVaR
                - Omega Ratio
                """
            )

        st.markdown("---")

        st.subheader("📈 Tickers")
        st.caption(f"{len(PORTFOLIO_TICKERS)} assets available")

        with st.expander("View All Tickers"):
            st.text("\n".join(PORTFOLIO_TICKERS))

        st.markdown("---")

        st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))


def main():
    """Main dashboard function."""
    render_header()
    render_sidebar()

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Latest Forecasts",
            "Portfolio Allocations",
            "Model Comparison",
            "🧪 Interactive",
        ]
    )

    with tab1:
        render_overview()

    with tab2:
        render_latest_forecasts()

    with tab3:
        render_portfolio_allocations()

    with tab4:
        render_model_comparison()

    with tab5:
        render_interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Dashboard error: {e}", exc_info=True)
