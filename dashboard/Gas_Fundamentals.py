import sys
from pathlib import Path
import io
import os

# Add src to Python path to avoid ModuleNotFoundError
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from gas_forecast.pipelines.balance import run_balance_pipeline
from gas_forecast.data.regions import (
    region_label,
    region_slug,
    supported_storage_regions,
)
from gas_forecast.llm.explain import generate_weekly_market_report, answer_market_question
from gas_forecast.modeling import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    sklearn_model_configs,
    ExpandingWindowSplitter,
    RecursiveForecaster,
    run_recursive_backtest,
)


@st.cache_data
def load_model_document() -> str:
    """Read the current gas model specification (disk I/O cached)."""
    doc_path = Path(__file__).resolve().parent.parent / "docs" / "models.md"
    if doc_path.exists():
        try:
            return doc_path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error loading model specification: {exc}"
    return "Model specification not found at `docs/models.md`."


st.set_page_config(
    page_title="Gas Fundamentals",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Gas Fundamentals")
st.subheader("Unified Weekly Supply-Demand Balance Model & LLM Price Explainability")

# Sidebar
st.sidebar.header("Configuration")
region_options = {
    region: region_label(region)
    for region in supported_storage_regions()
}
selected_region = st.sidebar.selectbox(
    "Select EIA Region",
    options=list(region_options.keys()),
    format_func=lambda x: region_options[x]
)

slug = region_slug(selected_region)
processed_dir = Path("datasets/processed")
balance_file = processed_dir / f"{slug}_weekly_supply_demand_balance_latest.parquet"

# Button to trigger balance pipeline
if st.sidebar.button("Run Pipeline / Refresh Data"):
    with st.spinner(f"Running supply-demand pipeline for {region_options[selected_region]}..."):
        try:
            saved_path = run_balance_pipeline(
                selected_region,
                processed_dir=processed_dir,
                force_refresh=True,
            )
            st.sidebar.success(f"Pipeline completed! Saved to {saved_path.name}")
        except Exception as e:
            st.sidebar.error(f"Pipeline failed: {e}")

if not balance_file.exists():
    st.warning(f"No processed balance sheet found for {region_options[selected_region]}. Please click the 'Run Pipeline / Refresh Data' button in the sidebar to fetch data and fit models.")
else:
    # Load data
    df = pd.read_parquet(balance_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if "duoarea" not in df.columns or not df["duoarea"].eq(selected_region).all():
        st.error(
            "The saved balance artifact does not match the selected EIA region. "
            "Refresh this region before using the dashboard."
        )
        st.stop()

    features_file = processed_dir / f"{slug}_weekly_model_features_latest.parquet"
    features_df = None
    if features_file.exists():
        features_df = pd.read_parquet(features_file)
        features_df["date"] = pd.to_datetime(features_df["date"])
        if not features_df["duoarea"].eq(selected_region).all():
            st.error(
                "The saved feature artifact does not match the selected EIA region. "
                "Refresh this region before forecasting."
            )
            st.stop()
        features_df = features_df.sort_values("date").reset_index(drop=True)

    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else latest_row

    # Metrics Section
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        is_national = (selected_region == "R48")
        price_col = "price" if (is_national or "regional_price" not in df.columns) else "regional_price"
        price_label = "Henry Hub Spot Price" if is_national else f"Regional Spot Price"
        price_diff = latest_row[price_col] - prev_row[price_col]
        st.metric(
            label=price_label,
            value=f"${latest_row[price_col]:.3f} / MMBtu",
            delta=f"${price_diff:+.3f}"
        )
    with col2:
        st.metric(
            label="Weekly Storage Change",
            value=f"{latest_row['weekly_change_bcf']:.1f} Bcf",
            delta=f"{(latest_row['weekly_change_bcf'] - prev_row['weekly_change_bcf']):+.1f} Bcf"
        )
    with col3:
        st.metric(
            label="Dry Gas Production",
            value=f"{latest_row['dry_production']:.1f} Bcf",
            delta=f"{(latest_row['dry_production'] - prev_row['dry_production']):+.1f} Bcf"
        )
    with col4:
        st.metric(
            label="Calculated Local Balance",
            value=f"{latest_row['local_balance']:.1f} Bcf",
            delta=f"{(latest_row['local_balance'] - prev_row['local_balance']):+.1f} Bcf")

    with st.expander("How the gas fundamentals model works", expanded=False):
        st.markdown(
            """
            1. **Build weekly regional supply and demand:** monthly EIA state data are
               disaggregated to storage weeks using daily weather, calendar effects, and spot prices.
            2. **Calculate the local balance:**
               `dry production - residential/commercial - power burn - industrial - fuel use`.
            3. **Reconcile with storage:** actual EIA storage change is compared with the local balance;
               the difference is the implied net inflow and balancing residual.
            4. **Forecast weekly storage change:** the selected model uses calendar cycles, HDD/CDD,
               lagged weather, lagged storage changes, and inventory relative to historical norms.
            5. **Roll storage forward:** each predicted weekly change is added to the previous projected
               storage level. Multiweek forecasts are recursive, so earlier predictions feed later weeks.
            6. **Backtest without leakage:** every fold trains only on weeks available before its validation window.

            Volumes are weekly Bcf unless otherwise labeled. The balance sheet explains physical context;
            the storage forecast is trained on leakage-safe model features rather than contemporaneous
            balance values that would not yet be known.
            """
        )

    forecast_tab, balance_tab, inventory_tab, flows_tab, prices_tab, diagnostics_tab = st.tabs([
        "1 · Forecast",
        "2 · Physical balance",
        "3 · Inventory / adequacy",
        "4 · Fuel & flows",
        "5 · Market prices",
        "6 · Diagnostics & models",
    ])

    with balance_tab:
        st.caption(
            "Weekly physical context. Dry production is supply; residential/commercial, power, "
            "industrial, and lease/plant/pipeline fuel are demand components estimated from EIA data."
        )
        st.markdown("### Weekly Supply and Demand Components (Bcf)")
        
        fig = go.Figure()
        # Production
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["dry_production"],
            name="Dry Production", mode="lines",
            line=dict(width=2.5, color="#10B981")
        ))
        # Consumption components as stacked area on demand side (visualized as positive for comparison)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["res_com"],
            name="Residential & Commercial", mode="lines",
            stackgroup="demand", line=dict(width=0.5, color="#EF4444")
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["power_burn"],
            name="Power Burn", mode="lines",
            stackgroup="demand", line=dict(width=0.5, color="#F59E0B")
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["industrial"],
            name="Industrial", mode="lines",
            stackgroup="demand", line=dict(width=0.5, color="#3B82F6")
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["fuel_use"],
            name="Lease/Plant/Pipeline Fuel", mode="lines",
            stackgroup="demand", line=dict(width=0.5, color="#8B5CF6")
        ))
        
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Volume (Bcf)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with inventory_tab:
        st.caption(
            "Inventory reconciliation. Local balance equals production minus modeled consumption; "
            "the residual equals actual EIA storage change minus that local balance."
        )
        st.markdown("### Local Balance vs Actual Weekly Storage Change")
        st.write("A negative local balance (consumption exceeds production) correlates with storage withdrawals (negative changes).")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["weekly_change_bcf"],
            name="Actual Storage Change", mode="lines",
            line=dict(width=2.0, color="#FF007F")
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["local_balance"],
            name="Local S/D Balance (Prod - Cons)", mode="lines",
            line=dict(width=2.0, color="#3B82F6", dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["net_inflow_balancing"],
            name="Inflow & Balancing Residual", mode="lines",
            line=dict(width=1.0, color="#9CA3AF")
        ))
        
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Volume (Bcf)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with prices_tab:
        st.caption(
            "Market-price context only. Daily Henry Hub observations are averaged into storage weeks; "
            "regional views add the available regional price series. This chart is not itself a price forecast."
        )
        is_national = (selected_region == "R48")
        title_text = "Henry Hub Spot Prices ($/MMBtu)" if is_national else f"Spot Price Comparison: Henry Hub vs. {region_options[selected_region]} ($/MMBtu)"
        st.markdown(f"### {title_text}")
        
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df["date"],
            y=df["price"],
            name="Henry Hub (Benchmark)",
            line=dict(color="#F59E0B", width=2)
        ))
        if not is_national and "regional_price" in df.columns:
            fig_price.add_trace(go.Scatter(
                x=df["date"],
                y=df["regional_price"],
                name=f"{region_options[selected_region]} (Basis-Adjusted)",
                line=dict(color="#3B82F6", width=2)
            ))
            
        fig_price.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Price ($/MMBtu)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with forecast_tab:
        st.caption(
            "Primary gas model. It predicts weekly EIA storage change, recursively adds each change to "
            "inventory, and evaluates historical accuracy with expanding-window backtests."
        )
        st.markdown("### Multi-Horizon Storage Forecasting")
        if features_df is None:
            st.warning("No weekly model features file found for this region. Please run the features stage of the data pipeline first.")
        else:
            # Layout columns for controls
            c1, c2, c3 = st.columns(3)
            with c1:
                sklearn_configs = {config.key: config for config in sklearn_model_configs()}
                selected_model_key = st.selectbox(
                    "Forecasting Model",
                    options=list(sklearn_configs.keys()),
                    format_func=lambda x: sklearn_configs[x].label
                )
            with c2:
                horizon_choice = st.selectbox(
                    "Forecast Horizon",
                    options=["1 Week", "4 Weeks", "End of Season"]
                )
            with c3:
                temp_anomaly = st.slider(
                    "Temperature Anomaly Scenario (°F)",
                    min_value=-5.0, max_value=5.0, value=0.0, step=0.5,
                    help="Shift the baseline weather forecast temperature up or down to test scenarios."
                )

            # Fit the forecast model on features available from the storage and
            # weather pipeline. The balance sheet is retained for analysis,
            # not treated as an as-of forecast input.
            feature_cols = list(DEFAULT_FEATURE_COLUMNS)
            target_col = DEFAULT_TARGET_COLUMN
            
            # Clean and fit
            train_clean = features_df.dropna(subset=[target_col, *feature_cols])
            model_config = sklearn_configs[selected_model_key]
            fitted_model = model_config.build()
            fitted_model.fit(train_clean[feature_cols], train_clean[target_col])
            
            # Determine projection timeline
            last_actual_date = train_clean["date"].max()
            
            # Horizon weeks calculation
            if horizon_choice == "1 Week":
                H = 1
            elif horizon_choice == "4 Weeks":
                H = 4
            else:
                # End of Season target calculation
                last_month = last_actual_date.month
                if 4 <= last_month <= 10:
                    # Injection season ends last Friday of October
                    end_season = pd.Timestamp(year=last_actual_date.year, month=10, day=31)
                else:
                    # Withdrawal season ends last Friday of March
                    year = last_actual_date.year if last_month in [1, 2, 3] else last_actual_date.year + 1
                    end_season = pd.Timestamp(year=year, month=3, day=31)
                
                # Calculate number of weeks remaining (rounded up)
                H = ((end_season - last_actual_date).days + 6) // 7
            
            # Generate predictions
            forecaster = RecursiveForecaster(
                fitted_model,
                feature_cols,
                model_key=selected_model_key,
            )
            proj = forecaster.predict_horizon(
                features_df=features_df,
                start_date=last_actual_date + pd.Timedelta(weeks=1),
                horizon_weeks=H,
                temp_anomaly=temp_anomaly
            )
            
            if proj.empty:
                st.error("Failed to generate forecast projection.")
            else:
                # Plot the path
                # Combine historical actual storage with projected storage
                # Get last 20 weeks of history for context
                hist_context = features_df[features_df["date"] <= last_actual_date].tail(20)
                
                fig_f = go.Figure()
                # Actual Storage
                fig_f.add_trace(go.Scatter(
                    x=hist_context["date"], y=hist_context["storage_bcf"],
                    name="Actual Historical Storage", mode="lines+markers",
                    line=dict(color="#10B981", width=2.5)
                ))
                # Projected Storage path
                proj_dates = [last_actual_date] + proj["date"].tolist()
                proj_values = [hist_context.iloc[-1]["storage_bcf"]] + proj["projected_storage"].tolist()
                fig_f.add_trace(go.Scatter(
                    x=proj_dates, y=proj_values,
                    name=f"Projected Storage ({horizon_choice})", mode="lines+markers",
                    line=dict(color="#3B82F6", width=2.5, dash="dash")
                ))
                
                # Add horizontal line for historical 5-year average end of season level if projecting to end of season
                if horizon_choice == "End of Season":
                    eos_row = features_df[features_df["date"] == proj["date"].iloc[-1]]
                    if not eos_row.empty:
                        eos_5yr = eos_row.iloc[0]["storage_5yr_avg"]
                        fig_f.add_hline(
                            y=eos_5yr,
                            line_dash="dot",
                            line_color="#FF007F",
                            annotation_text=f"5-Year Avg Season End ({eos_5yr:.0f} Bcf)"
                        )
                
                fig_f.update_layout(
                    template="plotly_dark",
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title="Total Storage (Bcf)",
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_f, use_container_width=True)
                
                # KPI metrics for the forecast
                latest_proj_val = proj.iloc[-1]["projected_storage"]
                latest_proj_date = proj.iloc[-1]["date"].strftime("%Y-%m-%d")
                
                # Get comparison average for that final date
                final_row = features_df[features_df["date"] == proj.iloc[-1]["date"]]
                avg_diff_str = ""
                if not final_row.empty:
                    final_5yr = final_row.iloc[0]["storage_5yr_avg"]
                    diff = latest_proj_val - final_5yr
                    avg_diff_str = f" ({diff:+.1f} Bcf vs 5-Year Avg)"
                    
                st.info(f"👉 **Projected Storage Level on {latest_proj_date}: {latest_proj_val:.1f} Bcf{avg_diff_str}**")

                # ── Weekly Storage Change: Forecast vs Actuals ───────────────────────
                st.markdown("#### Weekly Storage Change: Forecast vs. Actuals")
                st.caption(
                    "The blue dashed line shows the model's week-ahead predicted storage change. "
                    "The green line shows realized weekly changes from the EIA for the same history window. "
                    "In the forward projection window, only the forecast line is shown (no actuals available yet)."
                )

                # Actual weekly changes over the history context window
                hist_changes = features_df[features_df["date"] <= last_actual_date].tail(20)

                fig_wc = go.Figure()
                fig_wc.add_trace(go.Scatter(
                    x=hist_changes["date"],
                    y=hist_changes["weekly_change_bcf"],
                    name="Actual Weekly Change",
                    mode="lines+markers",
                    line=dict(color="#10B981", width=2.5),
                    marker=dict(size=5),
                ))
                # Forecast weekly changes (forward window only)
                fig_wc.add_trace(go.Scatter(
                    x=proj["date"],
                    y=proj["predicted_weekly_change"],
                    name=f"Forecast Weekly Change ({horizon_choice})",
                    mode="lines+markers",
                    line=dict(color="#3B82F6", width=2.5, dash="dash"),
                    marker=dict(size=5, symbol="diamond"),
                ))
                # Vertical marker at the forecast start
                fig_wc.add_vline(
                    x=last_actual_date,
                    line_dash="dot",
                    line_color="#9CA3AF",
                    annotation_text="Forecast Start",
                    annotation_position="top left",
                )
                fig_wc.update_layout(
                    template="plotly_dark",
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title="Weekly Change (Bcf)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_wc, use_container_width=True)

                # Backtesting Section
                st.markdown("---")
                st.markdown("### 🔄 Rolling-Origin Backtesting Suite")
                st.write("Simulate model performance historically on multiple folds without data leakage. The model is trained strictly on historical data prior to each validation window.")

                bt_col1, bt_col2 = st.columns(2)
                with bt_col1:
                    n_folds = st.slider(
                        "Number of Backtest Folds",
                        min_value=2, max_value=12, value=4, step=1,
                        help="More folds = wider coverage of history but shorter training windows per fold."
                    )
                with bt_col2:
                    val_weeks_choice = st.selectbox(
                        "Validation Window per Fold",
                        options=[4, 8, 12, 26],
                        index=2,
                        format_func=lambda w: f"{w} weeks (~{w//4} months)",
                        help="How many weeks ahead each fold evaluates."
                    )

                if st.button("Run Aggregated Folds Backtest"):
                    with st.spinner(f"Running {n_folds}-fold recursive backtest..."):
                        try:
                            total_weeks = len(train_clean)
                            # Spread folds evenly across available history
                            step_weeks = max(val_weeks_choice, (total_weeks - val_weeks_choice) // n_folds)
                            min_train_end = train_clean["date"].max() - pd.Timedelta(weeks=n_folds * step_weeks)

                            splitter = ExpandingWindowSplitter(
                                date_col="date",
                                initial_train_start=train_clean["date"].min(),
                                initial_train_end=min_train_end,
                                val_weeks=val_weeks_choice,
                                step_weeks=step_weeks,
                            )
                            
                            backtest_preds, backtest_metrics = run_recursive_backtest(
                                df=train_clean,
                                feature_cols=feature_cols,
                                target_col=target_col,
                                date_col="date",
                                model=fitted_model,
                                splitter=splitter,
                                horizon_weeks=4
                            )
                            
                            # Display metrics table
                            st.write("**Aggregated Metrics by Forecast Horizon (Weeks Ahead)**")
                            st.table(backtest_metrics.rename(columns={
                                "horizon_weeks_ahead": "Horizon (Weeks)",
                                "mae": "MAE (Bcf)",
                                "rmse": "RMSE (Bcf)",
                                "bias": "Bias (Bcf)",
                                "n_samples": "Number of Samples"
                            }))

                            # ── Backtest visual: forecast vs actuals ─────────────────
                            st.markdown("#### Backtest: Forecast vs. Actual Weekly Storage Change")
                            st.caption(
                                "Each fold's predicted weekly changes (dashed) are overlaid against "
                                "realized EIA actuals (solid). Folds are colour-coded. "
                                "Divergence between the two lines indicates model error."
                            )
                            bt_valid = backtest_preds.dropna(subset=["actual_weekly_change"])
                            fold_colors = ["#3B82F6", "#F59E0B", "#A855F7", "#EC4899", "#14B8A6"]
                            fig_bt = go.Figure()
                            for fold_id, fold_df in bt_valid.groupby("fold"):
                                color = fold_colors[(fold_id - 1) % len(fold_colors)]
                                fig_bt.add_trace(go.Scatter(
                                    x=fold_df["date"],
                                    y=fold_df["actual_weekly_change"],
                                    name=f"Actual – Fold {fold_id}",
                                    mode="lines+markers",
                                    line=dict(color=color, width=2),
                                    marker=dict(size=5),
                                    legendgroup=f"fold{fold_id}",
                                ))
                                fig_bt.add_trace(go.Scatter(
                                    x=fold_df["date"],
                                    y=fold_df["predicted_weekly_change"],
                                    name=f"Forecast – Fold {fold_id}",
                                    mode="lines+markers",
                                    line=dict(color=color, width=2, dash="dash"),
                                    marker=dict(size=5, symbol="diamond"),
                                    legendgroup=f"fold{fold_id}",
                                ))
                            fig_bt.update_layout(
                                template="plotly_dark",
                                hovermode="x unified",
                                xaxis_title="Date",
                                yaxis_title="Weekly Change (Bcf)",
                                margin=dict(l=20, r=20, t=30, b=20),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            )
                            st.plotly_chart(fig_bt, use_container_width=True)

                            st.success("Backtest completed successfully! Training sets were strictly isolated from test sets.")
                        except Exception as e:
                            st.error(f"Backtest failed: {e}")


    with flows_tab:
        st.caption(
            "Derived market context. Tightness compares the local balance with its contemporaneous seasonal norm; "
            "border/pipeline flow is the residual needed to reconcile local balance with actual storage change."
        )
        st.markdown("### Deeper Market Dynamics & Flows")
        
        # Calculate Market Tightness
        df = df.sort_values("date")
        df["seasonal_balance_norm"] = (
            df.groupby("week_of_year")["local_balance"]
            .transform(lambda x: x.rolling(5, min_periods=1, center=True).mean())
        )
        df["market_tightness"] = df["local_balance"] - df["seasonal_balance_norm"]
        df["market_tightness_rolling"] = df["market_tightness"].rolling(window=5, min_periods=1).mean()
        
        st.write("**Market Tightness Indicator (Bcf/week)**")
        st.caption("Measures how oversupplied (positive, loose/bearish) or undersupplied (negative, tight/bullish) the market is relative to the contemporaneous 5-year centered rolling average for that week of the year. This removes the long-term multi-year shale production growth trend.")
        st.info("💡 **Methodology Note**: Rather than comparing weekly balances to a global 16-year baseline average (which would misclassify all early years as undersupplied and all later years as oversupplied due to the doubling of U.S. shale production), we apply a **5-year centered rolling seasonal norm**. This dynamically adjusts the reference baseline to match the contemporaneous market size of each era.")
        
        fig_mt = go.Figure()
        # Bar colors: green for tight (negative balance deviation), red for loose (positive balance deviation)
        bar_colors = ["#EF4444" if val >= 0 else "#10B981" for val in df["market_tightness"]]
        fig_mt.add_trace(go.Bar(
            x=df["date"], y=df["market_tightness"],
            name="Weekly Deviation",
            marker_color=bar_colors,
            opacity=0.6
        ))
        fig_mt.add_trace(go.Scatter(
            x=df["date"], y=df["market_tightness_rolling"],
            name="5-Week Rolling Trend",
            line=dict(color="#3B82F6", width=2.5)
        ))
        fig_mt.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Volume Deviation (Bcf)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_mt, use_container_width=True)
        
        # Calculate Net Border / Pipeline Flows
        # Flow = local_balance - weekly_change_bcf
        df["net_border_flows"] = df["local_balance"] - df["weekly_change_bcf"]
        df["net_border_flows_rolling"] = df["net_border_flows"].rolling(window=12, min_periods=1).mean()
        
        st.markdown("---")
        st.write("**Estimated Net Border & Pipeline Flows (Bcf/week)**")
        st.caption("Calculated as the physical residual: `Local Balance - Actual Storage Change`. For the national Lower 48 model, a positive value represents Net Exports (outflows like LNG cargo exports and net pipeline exports to Mexico and Canada), while a negative value represents Net Imports (inflows). For regional models, this also includes inter-regional flows.")
        
        fig_bf = go.Figure()
        fig_bf.add_trace(go.Scatter(
            x=df["date"], y=df["net_border_flows"],
            name="Weekly Net Flows", mode="lines",
            line=dict(color="#FF007F", width=1.0),
            opacity=0.3
        ))
        fig_bf.add_trace(go.Scatter(
            x=df["date"], y=df["net_border_flows_rolling"],
            name="12-Week Rolling Average", mode="lines",
            line=dict(color="#10B981", width=2.5)
        ))
        fig_bf.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Net Inflow (+)/Outflow (-) (Bcf)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_bf, use_container_width=True)

    with diagnostics_tab:
        st.caption(
            "Model documentation and assumptions. Forecast accuracy controls live under the Forecast tab "
            "because they evaluate the storage model shown there."
        )
        st.markdown("## 📖 Rigorous Mathematical Methodology & Framework")
        st.markdown(load_model_document())
    st.markdown("---")
    st.header("🧠 Gemini Price Explainability Layer")
    st.write("Select a week to generate a natural language market analysis report explaining the supply/demand drivers and spot price impact.")

    # Dropdown with descending order dates
    available_dates = df["date"].sort_values(ascending=False).unique()
    selected_date = st.selectbox(
        "Select Week Ending Date",
        options=available_dates,
        format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d (%A)")
    )

    if st.button("Generate Market Analysis Report"):
        with st.spinner("Calling Gemini API to analyze market balance and price movements..."):
            try:
                report = generate_weekly_market_report(
                    balance_df=df,
                    target_date=selected_date,
                    region_name=region_options[selected_region],
                )
                st.markdown("### Market Analysis Report")
                st.markdown(report)
            except Exception as e:
                st.error(f"Failed to generate report: {e}")

    # Chat / Ask anything Section
    st.markdown("---")
    st.markdown("### 💬 Ask Gemini about the Models, Data, or Analysis")
    st.caption("Ask custom questions about this week's supply-demand dynamics, OLS regression equations, downscaling logic, or forecasting models.")
    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., Why did power burn increase this week? or Explain the pipeline fuel downscaling logic."
    )
    if st.button("Ask Gemini"):
        if not user_question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Calling Gemini API to answer your question..."):
                try:
                    answer = answer_market_question(
                        balance_df=df,
                        target_date=selected_date,
                        question=user_question,
                        region_name=region_options[selected_region],
                    )
                    st.markdown("### Gemini Answer")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Failed to get answer: {e}")
