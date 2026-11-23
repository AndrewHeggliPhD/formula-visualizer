"""Interactive Plotly charts for statistical distributions and custom formulas."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from src.i18n import t

PRIMARY_BLUE = "#2563EB"
PRIMARY_FILL = "rgba(37, 99, 235, 0.14)"
GHOST_COLOR = "#94A3B8"
FOCUS_COLOR = "#F59E0B"
MEAN_COLOR = "#7C3AED"


def build_distribution_chart(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    kind: str,
    view_type: str,
    x_label: str,
    y_label: str,
    title: str,
    focus_x: float,
    focus_y: float,
    mean_val: Optional[float] = None,
    ghost_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    lang: str = "no",
) -> go.Figure:
    """Create an interactive Plotly chart for a continuous or discrete distribution."""
    fig = go.Figure()

    # Optional Ghost reference curve
    if ghost_data is not None:
        gx, gy = ghost_data
        if kind == "discrete" and view_type != "cdf":
            fig.add_trace(
                go.Bar(
                    x=gx,
                    y=gy,
                    name=t("chart_ghost", lang),
                    marker_color="rgba(148, 163, 184, 0.35)",
                    marker_line_color=GHOST_COLOR,
                    marker_line_width=1.2,
                    hovertemplate=f"{x_label}=%{{x}}<br>{y_label}=%{{y:.4g}}<extra></extra>",
                )
            )
        else:
            line_shape = "hv" if (kind == "discrete" and view_type == "cdf") else "linear"
            fig.add_trace(
                go.Scatter(
                    x=gx,
                    y=gy,
                    mode="lines",
                    name=t("chart_ghost", lang),
                    line=dict(color=GHOST_COLOR, width=2.2, dash="dash", shape=line_shape),
                    hovertemplate=f"{x_label}=%{{x:.3g}}<br>{y_label}=%{{y:.4g}}<extra></extra>",
                )
            )

    # Current distribution trace
    if kind == "discrete" and view_type != "cdf":
        bar_colors = [
            FOCUS_COLOR if abs(float(xv) - round(focus_x)) < 0.5 else PRIMARY_BLUE
            for xv in x_grid
        ]
        fig.add_trace(
            go.Bar(
                x=x_grid,
                y=y_grid,
                name=t("chart_current", lang),
                marker_color=bar_colors,
                opacity=0.88,
                hovertemplate=f"{x_label}=%{{x}}<br>{y_label}=%{{y:.4g}}<extra></extra>",
            )
        )
    else:
        line_shape = "hv" if (kind == "discrete" and view_type == "cdf") else "linear"
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                name=t("chart_current", lang),
                line=dict(color=PRIMARY_BLUE, width=3.0, shape=line_shape),
                fill="tozeroy" if view_type != "cdf" else None,
                fillcolor=PRIMARY_FILL,
                hovertemplate=f"{x_label}=%{{x:.3g}}<br>{y_label}=%{{y:.4g}}<extra></extra>",
            )
        )

    # Mean reference line
    if mean_val is not None and math.isfinite(mean_val):
        x_min, x_max = float(np.nanmin(x_grid)), float(np.nanmax(x_grid))
        if x_min <= mean_val <= x_max:
            fig.add_vline(
                x=mean_val,
                line_width=1.6,
                line_dash="dot",
                line_color=MEAN_COLOR,
                annotation_text=f"{t('chart_mean', lang)} = {mean_val:.3g}",
                annotation_position="top right",
                annotation_font_color=MEAN_COLOR,
            )

    # Focus point marker & vertical guide
    if math.isfinite(focus_x) and math.isfinite(focus_y):
        fig.add_vline(
            x=focus_x,
            line_width=1.8,
            line_dash="dash",
            line_color=FOCUS_COLOR,
        )
        fig.add_trace(
            go.Scatter(
                x=[focus_x],
                y=[focus_y],
                mode="markers+text",
                name=f"{t('chart_focus', lang)} ({focus_x:.3g})",
                marker=dict(
                    color=FOCUS_COLOR,
                    size=13,
                    line=dict(color="#FFFFFF", width=2.5),
                ),
                text=[f"({focus_x:.3g}, {focus_y:.4g})"],
                textposition="top center",
                textfont=dict(color="#B45309", size=12),
                hovertemplate=f"{t('chart_focus', lang)}<br>{x_label}=%{{x:.4g}}<br>{y_label}=%{{y:.5g}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#0F172A")),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=410,
        margin=dict(l=45, r=25, t=48, b=45),
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", rangemode="tozero")
    return fig


def build_custom_curve_chart(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    is_discrete: bool,
    x_label: str,
    y_label: str,
    current_x: float,
    current_y: float,
    ghost_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    lang: str = "no",
) -> go.Figure:
    """Create an interactive 2D chart for a custom formula (function or scalar sweep)."""
    fig = go.Figure()

    # Filter extreme spikes so singular asymptotes (like 1/a near 0) still read cleanly
    finite_mask = np.isfinite(y_grid)
    y_plot = np.where(finite_mask, y_grid, np.nan)

    if ghost_data is not None:
        gx, gy = ghost_data
        gy_plot = np.where(np.isfinite(gy), gy, np.nan)
        if is_discrete:
            fig.add_trace(
                go.Bar(
                    x=gx,
                    y=gy_plot,
                    name=t("chart_ghost", lang),
                    marker_color="rgba(148, 163, 184, 0.35)",
                    marker_line_color=GHOST_COLOR,
                    marker_line_width=1.2,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=gx,
                    y=gy_plot,
                    mode="lines",
                    name=t("chart_ghost", lang),
                    line=dict(color=GHOST_COLOR, width=2.2, dash="dash"),
                )
            )

    if is_discrete:
        bar_colors = [
            FOCUS_COLOR if abs(float(xv) - round(current_x)) < 0.5 else PRIMARY_BLUE
            for xv in x_grid
        ]
        fig.add_trace(
            go.Bar(
                x=x_grid,
                y=y_plot,
                name=t("chart_current", lang),
                marker_color=bar_colors,
                opacity=0.88,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_plot,
                mode="lines",
                name=t("chart_current", lang),
                line=dict(color=PRIMARY_BLUE, width=3.0),
                hovertemplate=f"{x_label}=%{{x:.4g}}<br>{y_label}=%{{y:.5g}}<extra></extra>",
            )
        )

    if math.isfinite(current_x) and math.isfinite(current_y):
        fig.add_vline(
            x=current_x,
            line_width=1.8,
            line_dash="dash",
            line_color=FOCUS_COLOR,
        )
        fig.add_trace(
            go.Scatter(
                x=[current_x],
                y=[current_y],
                mode="markers+text",
                name=f"{t('chart_focus', lang)} ({current_x:.3g})",
                marker=dict(
                    color=FOCUS_COLOR,
                    size=13,
                    line=dict(color="#FFFFFF", width=2.5),
                ),
                text=[f"({current_x:.3g}, {current_y:.4g})"],
                textposition="top center",
                textfont=dict(color="#B45309", size=12),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{t('chart_response_title', lang)}: {y_label} vs. {x_label}",
            font=dict(size=16, color="#0F172A"),
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=390,
        margin=dict(l=45, r=25, t=48, b=45),
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    return fig
