from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from energy_forecast.artifacts import save_versioned_parquet
from oil_forecast.backtesting import run_oil_backtest
from oil_forecast.pipelines import (
    build_oil_forecast,
    load_oil_balance,
    refresh_oil_data,
)


PROCESSED_DIR = ROOT / "datasets" / "processed"
CACHE_DIR = ROOT / "datasets" / "cache" / "oil"
BALANCE_PATH = PROCESSED_DIR / "us_weekly_crude_balance_latest.parquet"
FORECAST_PATH = PROCESSED_DIR / "us_weekly_crude_forecast_latest.parquet"
PREDICTIONS_PATH = (
    PROCESSED_DIR / "us_weekly_crude_backtest_predictions_latest.parquet"
)
METRICS_PATH = PROCESSED_DIR / "us_weekly_crude_backtest_metrics_latest.parquet"


def _materialize_model_outputs(initial_train_weeks: int) -> None:
    balance = load_oil_balance(PROCESSED_DIR)
    build_oil_forecast(processed_dir=PROCESSED_DIR)
    predictions, metrics = run_oil_backtest(
        balance,
        initial_train_weeks=initial_train_weeks,
    )
    save_versioned_parquet(
        predictions,
        PROCESSED_DIR,
        "us_weekly_crude_backtest_predictions",
    )
    save_versioned_parquet(
        metrics,
        PROCESSED_DIR,
        "us_weekly_crude_backtest_metrics",
    )


def _change_label(value: float) -> str:
    direction = "draw" if value < 0 else "build"
    return f"{abs(value):.2f} MMbbl {direction}"


st.set_page_config(page_title="Oil Fundamentals", page_icon="🛢️", layout="wide")
st.title("🛢️ U.S. Crude Oil Fundamentals")
st.caption(
    "Weekly commercial crude inventory forecast built from production, imports, "
    "refinery inputs, exports, SPR movement, and the historical balance adjustment."
)

st.sidebar.header("Oil forecast")
history_start = st.sidebar.date_input(
    "History start",
    value=date(2010, 1, 1),
    help="The EIA legacy-series route may return older rows; the adapter enforces this window locally.",
)
initial_train_weeks = int(
    st.sidebar.number_input(
        "Initial backtest training weeks",
        min_value=52,
        max_value=520,
        value=156,
        step=52,
    )
)

if st.sidebar.button("Refresh EIA data and rebuild", type="primary"):
    with st.spinner("Refreshing six weekly EIA crude series and rebuilding outputs..."):
        try:
            refresh_oil_data(
                start=history_start.isoformat(),
                cache_dir=CACHE_DIR,
                processed_dir=PROCESSED_DIR,
            )
            _materialize_model_outputs(initial_train_weeks)
            st.sidebar.success("Oil data, forecast, and backtest refreshed.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Refresh failed: {exc}")

if st.sidebar.button("Rebuild from local balance"):
    with st.spinner("Rebuilding forecast and backtest from local data..."):
        try:
            _materialize_model_outputs(initial_train_weeks)
            st.sidebar.success("Forecast and backtest rebuilt.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Rebuild failed: {exc}")

if not BALANCE_PATH.exists() or not FORECAST_PATH.exists():
    st.warning(
        "No oil forecast artifacts were found. Use the sidebar refresh, or run "
        "`python -m oil_forecast.cli refresh` followed by "
        "`python -m oil_forecast.cli forecast`."
    )
    st.stop()

balance = pd.read_parquet(BALANCE_PATH)
forecast = pd.read_parquet(FORECAST_PATH)
for frame in (balance, forecast):
    frame["date"] = pd.to_datetime(frame["date"])
balance = balance.sort_values("date").reset_index(drop=True)
forecast = forecast.sort_values("date").reset_index(drop=True)

balance_required = {
    "date",
    "commercial_stocks_mmbbl",
    "spr_stocks_mmbbl",
    "commercial_stock_change_mmbbl",
    "fundamental_balance_mmbbl",
    "balance_adjustment_mmbbl",
}
forecast_required = {
    "date",
    "prediction_mmbbl",
    "last_change_baseline_mmbbl",
    "production_forecast_mmbbl",
    "imports_forecast_mmbbl",
    "refinery_inputs_forecast_mmbbl",
    "exports_forecast_mmbbl",
    "spr_stock_change_forecast_mmbbl",
    "balance_adjustment_forecast_mmbbl",
}
missing_balance = sorted(balance_required - set(balance.columns))
missing_forecast = sorted(forecast_required - set(forecast.columns))
if missing_balance or missing_forecast:
    st.error(
        "Oil artifacts do not match the dashboard schema. "
        f"Missing balance columns: {missing_balance}; forecast columns: {missing_forecast}."
    )
    st.stop()

predictions = (
    pd.read_parquet(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else pd.DataFrame()
)
metrics = pd.read_parquet(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
if not predictions.empty:
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True)
    predictions["forecast_origin"] = pd.to_datetime(
        predictions["forecast_origin"], utc=True
    )
    predictions = predictions.sort_values("date").reset_index(drop=True)

latest = forecast.iloc[-1]
latest_balance = balance.iloc[-1]
latest_observation = pd.Timestamp(balance["date"].max()).normalize()
age_days = (pd.Timestamp.now().normalize() - latest_observation).days
if age_days > 14:
    st.warning(f"Latest EIA observation is {age_days} days old ({latest_observation:%Y-%m-%d}).")
else:
    st.success(f"Latest EIA week ending: {latest_observation:%Y-%m-%d}")

fundamentals_mae = baseline_mae = None
if not metrics.empty and {"model", "mae"}.issubset(metrics.columns):
    indexed_metrics = metrics.set_index("model")
    if {
        "seasonal_level_fundamentals",
        "last_change_baseline",
    }.issubset(indexed_metrics.index):
        fundamentals_mae = float(
            indexed_metrics.loc["seasonal_level_fundamentals", "mae"]
        )
        baseline_mae = float(indexed_metrics.loc["last_change_baseline", "mae"])

cards = st.columns(4)
cards[0].metric("Forecast week ending", pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"))
forecast_change = float(latest["prediction_mmbbl"])
cards[1].metric(
    "Forecast draw" if forecast_change < 0 else "Forecast build",
    f"{abs(forecast_change):.2f} MMbbl",
)
reported_change = float(latest_balance["commercial_stock_change_mmbbl"])
cards[2].metric(
    "Latest draw" if reported_change < 0 else "Latest build",
    f"{abs(reported_change):.2f} MMbbl",
)
if fundamentals_mae is not None and baseline_mae:
    improvement = 1.0 - fundamentals_mae / baseline_mae
    cards[3].metric("MAE vs baseline", f"{improvement:.1%} better")
else:
    cards[3].metric("Backtest", "Not materialized")

with st.expander("How the oil fundamentals model works", expanded=False):
    st.markdown(
        """
        1. **Load weekly EIA series** for production, imports, refinery inputs,
           exports, commercial crude stocks, and SPR stocks.
        2. **Convert daily flow rates to weekly million barrels.**
        3. **Build the physical balance:**
           `production + imports - refinery inputs - exports - SPR stock change`.
        4. **Calculate the historical adjustment:** reported commercial stock change
           minus the physical balance. It captures timing, transfers, rounding, and
           terms outside the six headline series.
        5. **Forecast each component** from its trailing five-year week-of-year profile
           plus a four-week level adjustment, using only earlier observations.
        6. **Recombine the forecasts through the balance identity** and compare the
           result with a last-reported-change baseline in expanding-window backtests.

        Negative inventory changes are draws; positive changes are builds. The current
        historical API is revised data, so the backtest is not a true publication-vintage replay.
        """
    )

forecast_tab, balance_tab, inventory_tab, backtest_tab, diagnostics_tab = st.tabs(
    [
        "1 · Forecast",
        "2 · Physical balance",
        "3 · Inventories",
        "4 · Backtest",
        "5 · Diagnostics",
    ]
)

with forecast_tab:
    st.subheader("Next-week balance decomposition")
    spr_contribution = -float(latest["spr_stock_change_forecast_mmbbl"])
    waterfall = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * 6 + ["total"],
            x=[
                "Production",
                "Imports",
                "Refinery inputs",
                "Exports",
                "SPR contribution",
                "Balance adjustment",
                "Inventory forecast",
            ],
            y=[
                float(latest["production_forecast_mmbbl"]),
                float(latest["imports_forecast_mmbbl"]),
                -float(latest["refinery_inputs_forecast_mmbbl"]),
                -float(latest["exports_forecast_mmbbl"]),
                spr_contribution,
                float(latest["balance_adjustment_forecast_mmbbl"]),
                float(latest["prediction_mmbbl"]),
            ],
            connector={"line": {"color": "#64748B"}},
            increasing={"marker": {"color": "#22C55E"}},
            decreasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#3B82F6"}},
        )
    )
    waterfall.update_layout(
        template="plotly_dark",
        yaxis_title="Million barrels per week",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(waterfall, width="stretch")
    st.caption(
        "SPR contribution is the negative of forecast SPR stock change: an SPR withdrawal "
        "adds barrels available to the commercial balance."
    )

    recent_changes = balance.tail(26)
    change_chart = go.Figure()
    change_chart.add_trace(
        go.Bar(
            x=recent_changes["date"],
            y=recent_changes["commercial_stock_change_mmbbl"],
            name="Reported change",
            marker_color="#94A3B8",
        )
    )
    change_chart.add_trace(
        go.Scatter(
            x=[latest["date"]],
            y=[latest["prediction_mmbbl"]],
            name="Fundamentals forecast",
            mode="markers",
            marker=dict(size=13, color="#3B82F6", symbol="diamond"),
        )
    )
    change_chart.add_trace(
        go.Scatter(
            x=[latest["date"]],
            y=[latest["last_change_baseline_mmbbl"]],
            name="Last-change baseline",
            mode="markers",
            marker=dict(size=11, color="#F59E0B", symbol="x"),
        )
    )
    change_chart.add_hline(y=0, line_color="#64748B")
    change_chart.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="Weekly change (MMbbl)",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(change_chart, width="stretch")

with balance_tab:
    st.subheader("Reported change versus modeled balance")
    years = st.selectbox("History window", [1, 3, 5, "All"], index=2)
    balance_view = balance
    if years != "All":
        cutoff = balance["date"].max() - pd.DateOffset(years=int(years))
        balance_view = balance.loc[balance["date"] >= cutoff]
    physical = go.Figure()
    for column, label, color, dash in (
        ("commercial_stock_change_mmbbl", "Reported stock change", "#F8FAFC", None),
        ("fundamental_balance_mmbbl", "Physical balance", "#3B82F6", "dash"),
        ("balance_adjustment_mmbbl", "Balance adjustment", "#F59E0B", "dot"),
    ):
        physical.add_trace(
            go.Scatter(
                x=balance_view["date"],
                y=balance_view[column],
                name=label,
                line=dict(color=color, width=2, dash=dash),
            )
        )
    physical.add_hline(y=0, line_color="#64748B")
    physical.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="Million barrels per week",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(physical, width="stretch")

with inventory_tab:
    st.subheader("Commercial and Strategic Petroleum Reserve inventories")
    inventory = go.Figure()
    inventory.add_trace(
        go.Scatter(
            x=balance["date"],
            y=balance["commercial_stocks_mmbbl"],
            name="Commercial crude excluding SPR",
            line=dict(color="#3B82F6", width=2.5),
        )
    )
    inventory.add_trace(
        go.Scatter(
            x=balance["date"],
            y=balance["spr_stocks_mmbbl"],
            name="SPR crude",
            line=dict(color="#A855F7", width=2.5),
        )
    )
    inventory.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="Million barrels",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(inventory, width="stretch")

with backtest_tab:
    if predictions.empty or metrics.empty:
        st.info("Use either sidebar rebuild button to materialize backtest outputs.")
    else:
        st.subheader("Expanding-window one-week backtest")
        st.dataframe(metrics, width="stretch", hide_index=True)
        backtest_view = predictions.tail(156)
        backtest_chart = go.Figure()
        if {
            "lower_bound_mmbbl",
            "upper_bound_mmbbl",
        }.issubset(backtest_view.columns) and backtest_view[
            "lower_bound_mmbbl"
        ].notna().any():
            backtest_chart.add_trace(
                go.Scatter(
                    x=backtest_view["date"],
                    y=backtest_view["upper_bound_mmbbl"],
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            backtest_chart.add_trace(
                go.Scatter(
                    x=backtest_view["date"],
                    y=backtest_view["lower_bound_mmbbl"],
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(59,130,246,0.16)",
                    name="80% conformal interval",
                )
            )
        for column, label, color, dash in (
            ("actual_mmbbl", "Actual", "#F8FAFC", None),
            ("prediction_mmbbl", "Fundamentals", "#3B82F6", None),
            ("last_change_baseline_mmbbl", "Last-change baseline", "#F59E0B", "dash"),
        ):
            backtest_chart.add_trace(
                go.Scatter(
                    x=backtest_view["date"],
                    y=backtest_view[column],
                    name=label,
                    line=dict(color=color, width=2, dash=dash),
                )
            )
        backtest_chart.add_hline(y=0, line_color="#64748B")
        backtest_chart.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            yaxis_title="Weekly change (MMbbl)",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(backtest_chart, width="stretch")

with diagnostics_tab:
    st.subheader("Calibration and data contract")
    diagnostic_cards = st.columns(4)
    diagnostic_cards[0].metric("Balance rows", f"{len(balance):,}")
    diagnostic_cards[1].metric(
        "Data window",
        f"{balance['date'].min():%Y}–{balance['date'].max():%Y}",
    )
    diagnostic_cards[2].metric("Backtest rows", f"{len(predictions):,}")
    interval_rows = pd.DataFrame()
    if not predictions.empty and {
        "lower_bound_mmbbl",
        "upper_bound_mmbbl",
        "actual_mmbbl",
    }.issubset(predictions.columns):
        interval_rows = predictions.dropna(
            subset=["lower_bound_mmbbl", "upper_bound_mmbbl", "actual_mmbbl"]
        )
    if not interval_rows.empty:
        covered = interval_rows["actual_mmbbl"].between(
            interval_rows["lower_bound_mmbbl"],
            interval_rows["upper_bound_mmbbl"],
        )
        coverage = float(covered.mean())
        width = float(
            (
                interval_rows["upper_bound_mmbbl"]
                - interval_rows["lower_bound_mmbbl"]
            ).mean()
        )
        diagnostic_cards[3].metric("80% interval coverage", f"{coverage:.1%}")
        st.caption(
            f"Eligible interval rows: {len(interval_rows):,}; average interval width: "
            f"{width:.2f} MMbbl. Coverage below 80% means the current intervals are too narrow."
        )
    else:
        diagnostic_cards[3].metric("Interval coverage", "Unavailable")

    st.markdown("#### Latest forecast fields")
    latest_fields = latest.to_frame(name="value").reset_index(names="field")
    latest_fields["value"] = latest_fields["value"].map(str)
    st.dataframe(latest_fields, width="stretch", hide_index=True)
    st.info(
        "Timing limitation: current EIA history contains revised values rather than every original "
        "Weekly Petroleum Status Report vintage. Target-week features are excluded, but this remains "
        "a revised-history diagnostic."
    )
