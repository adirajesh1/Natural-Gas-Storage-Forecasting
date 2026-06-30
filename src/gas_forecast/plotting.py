import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_DATE_COLUMN = "date"
DEFAULT_ACTUAL_COLUMN = "weekly_change_bcf"
DEFAULT_PREDICTED_COLUMN = "predicted_weekly_change"
DEFAULT_DEVIATION_COLUMN = "forecast_deviation"


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"Forecast data missing required columns: {sorted(missing)}")


def plot_weekly_change_forecast(
    forecast: pd.DataFrame,
    *,
    model_name: str = "Forecast",
    title: str | None = None,
    date_col: str = DEFAULT_DATE_COLUMN,
    actual_col: str = DEFAULT_ACTUAL_COLUMN,
    predicted_col: str = DEFAULT_PREDICTED_COLUMN,
    deviation_col: str = DEFAULT_DEVIATION_COLUMN,
) -> go.Figure:
    """Build the standard actual-vs-forecast comparison chart."""
    _require_columns(forecast, {date_col, actual_col, predicted_col})

    plot_data = forecast.copy()
    plot_data[date_col] = pd.to_datetime(plot_data[date_col])
    plot_data = plot_data.sort_values(date_col).reset_index(drop=True)
    if deviation_col not in plot_data.columns:
        plot_data[deviation_col] = plot_data[actual_col] - plot_data[predicted_col]

    has_bands = {"lower_band", "upper_band"}.issubset(plot_data.columns)
    band_label = "+/- 1 std range" if has_bands else None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=(
            f"Actual vs {model_name}" + (" (+/- 1 std)" if has_bands else ""),
            "Deviation from Forecast",
        ),
    )

    if has_bands:
        fig.add_trace(
            go.Scatter(
                x=plot_data[date_col],
                y=plot_data["upper_band"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_data[date_col],
                y=plot_data["lower_band"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(99, 110, 250, 0.2)",
                name=band_label,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=plot_data[date_col],
            y=plot_data[predicted_col],
            mode="lines",
            name=model_name,
            line=dict(color="#636efa", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_data[date_col],
            y=plot_data[actual_col],
            mode="lines+markers",
            name="Actual weekly change",
            line=dict(color="#EF553B", width=2.5),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )

    if has_bands and "outside_band" in plot_data.columns:
        outliers = plot_data[plot_data["outside_band"]]
        if not outliers.empty:
            fig.add_trace(
                go.Scatter(
                    x=outliers[date_col],
                    y=outliers[actual_col],
                    mode="markers",
                    name="Outside +/- 1 std",
                    marker=dict(size=10, color="#FFA15A", symbol="diamond"),
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Bar(
            x=plot_data[date_col],
            y=plot_data[deviation_col],
            name="Deviation",
            marker_color=[
                "#00CC96" if v >= 0 else "#AB63FA"
                for v in plot_data[deviation_col]
            ],
            opacity=0.85,
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=title or f"Weekly Storage Change vs {model_name}",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="BCF", row=1, col=1)
    fig.update_yaxes(title_text="Deviation (BCF)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig
