"""FormelVis / FormulaViz — Interactive Visual Explorer for Statistical Distributions & Formulas."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from src.charts import build_custom_curve_chart, build_distribution_chart
from src.distributions import (
    DISTRIBUTIONS,
    DistributionSpec,
    compute_distribution_stats,
    evaluate_distribution_curve,
    get_distribution_parsed,
)
from src.examples import EXAMPLE_FORMULAS, FormulaExample
from src.formula_engine import (
    EvaluationResult,
    ParsedFormula,
    evaluate_curve,
    evaluate_formula_point,
    parse_custom_formula,
)
from src.formula_renderer import render_breakdown_html, render_formula_card_html
from src.i18n import t

st.set_page_config(
    page_title="FormelVis / FormulaViz",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle global polish
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .fv-insight-box {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        border-radius: 0 10px 10px 0;
        padding: 10px 14px;
        font-size: 0.88rem;
        color: #1E3A8A;
        margin: 8px 0 14px 0;
    }
    .fv-scalar-badge {
        display: inline-block;
        background: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_distribution_parsed(dist_key: str) -> ParsedFormula:
    return get_distribution_parsed(DISTRIBUTIONS[dist_key])


@st.cache_resource(show_spinner=False)
def cached_custom_parsed(raw_formula: str) -> ParsedFormula:
    return parse_custom_formula(raw_formula)


def _init_session_state() -> None:
    """Initialize session state, applying optional URL query parameters on first load."""
    if st.session_state.get("_initialized"):
        return

    qp = st.query_params
    lang = qp.get("lang", "no")
    if lang not in ("no", "en"):
        lang = "no"
    st.session_state["lang"] = lang

    mode = qp.get("mode", "distributions")
    if mode not in ("distributions", "custom"):
        mode = "distributions"
    st.session_state["mode"] = mode

    dist_key = qp.get("dist", "normal")
    if dist_key not in DISTRIBUTIONS:
        dist_key = "normal"
    st.session_state["dist_key"] = dist_key

    default_custom = EXAMPLE_FORMULAS[0].formula
    formula_qp = qp.get("formula", "")
    st.session_state["custom_formula_text"] = formula_qp.strip() if formula_qp else default_custom

    st.session_state["active_example_key"] = EXAMPLE_FORMULAS[0].key if not formula_qp else ""
    st.session_state["saved_formulas"] = []
    st.session_state["ghost_dist"] = None
    st.session_state["ghost_custom"] = None
    st.session_state["baselines"] = {}
    st.session_state["custom_ranges"] = {}

    # Restore optional params JSON from URL
    params_raw = qp.get("params", "")
    if params_raw:
        try:
            loaded_params = json.loads(params_raw)
            if isinstance(loaded_params, dict):
                st.session_state["url_params_seed"] = {
                    str(k): float(v) for k, v in loaded_params.items()
                }
        except Exception:
            pass

    st.session_state["_initialized"] = True


def _format_metric(val: float) -> str:
    if not math.isfinite(val):
        return "—"
    if abs(val) == 0:
        return "0"
    if abs(val) >= 10000 or (0 < abs(val) < 0.0001):
        return f"{val:.3e}"
    return f"{val:.4g}"


def _format_delta(curr: float, ref: Optional[float]) -> Optional[str]:
    if ref is None or not (math.isfinite(curr) and math.isfinite(ref)):
        return None
    diff = curr - ref
    if abs(diff) < 1e-9:
        return "0"
    sign = "+" if diff > 0 else ""
    return f"{sign}{_format_metric(diff)}"


def _get_or_create_baseline(
    signature: str,
    parsed: ParsedFormula,
    initial_values: Dict[str, float],
) -> Dict[str, Any]:
    """Store baseline output, values, and partial derivatives for smooth visual scaling."""
    baselines: Dict[str, Dict[str, Any]] = st.session_state["baselines"]
    if signature not in baselines:
        probe = evaluate_formula_point(parsed, initial_values, scaling_mode="sensitivity")
        base_out = probe.output_value if (probe.valid and abs(probe.output_value) > 1e-12) else 1.0
        base_partials = {
            oid: occ.partial_derivative for oid, occ in probe.occurrences.items()
        }
        baselines[signature] = {
            "output": base_out,
            "values": dict(initial_values),
            "partials": base_partials,
        }
    return baselines[signature]


def _render_details_table(eval_res: EvaluationResult, lang: str) -> None:
    rows: List[Dict[str, Any]] = []
    for occ in eval_res.occurrences.values():
        meta = occ.meta
        ctx = meta.context_no if lang == "no" else meta.context_en
        sym_label = (
            f"{meta.display_name} (#{meta.occurrence_index})"
            if meta.total_occurrences > 1
            else meta.display_name
        )
        dir_key = {"pos": "dir_up", "neg": "dir_down", "neutral": "dir_neutral"}[occ.direction]
        rows.append(
            {
                t("col_symbol", lang): sym_label,
                t("col_context", lang): ctx,
                t("col_value", lang): _format_metric(occ.value),
                t("col_partial", lang): _format_metric(occ.partial_derivative),
                t("col_elasticity", lang): _format_metric(occ.elasticity),
                t("col_direction", lang): t(dir_key, lang),
                t("col_scale", lang): f"{occ.scale:.2f}×",
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _find_dominant_occurrence(eval_res: EvaluationResult, lang: str) -> str:
    items = list(eval_res.occurrences.values())
    if not items:
        return "—"
    top = max(items, key=lambda o: abs(o.elasticity))
    meta = top.meta
    if meta.total_occurrences > 1:
        ctx = meta.context_no if lang == "no" else meta.context_en
        return f"{meta.display_name} #{meta.occurrence_index} ({ctx})"
    return meta.display_name


def _get_example_by_formula(formula_text: str) -> Optional[FormulaExample]:
    norm = formula_text.strip()
    for ex in EXAMPLE_FORMULAS:
        if ex.formula.strip() == norm:
            return ex
    return None


def _default_range_for_var(
    var_name: str,
    display_name: str,
    default_val: float,
    matched_example: Optional[FormulaExample],
) -> Tuple[float, float, float]:
    """Determine sensible slider bounds (min, max, step) for a custom formula variable."""
    if matched_example and display_name in matched_example.ranges:
        return matched_example.ranges[display_name]
    lower = display_name.lower()
    if lower in {"p", "prob"} or lower.startswith("p("):
        return (0.01, 1.0, 0.01)
    if default_val >= 50:
        return (1.0, max(500.0, default_val * 4.0), 1.0)
    if default_val > 0:
        hi = max(10.0, default_val * 5.0)
        lo = 0.05 if lower in {"a", "sigma", "σ", "df", "df(t)", "n", "beta", "alpha", "lam"} else -hi
        return (lo, hi, round((hi - lo) / 100.0, 3) or 0.1)
    return (-10.0, 10.0, 0.1)


def _run_distributions_mode(lang: str, scaling_mode: str, color_coding: bool) -> Dict[str, float]:
    """Render sidebar controls and main canvas for built-in statistical distributions."""
    dist_keys = list(DISTRIBUTIONS.keys())
    current_key = st.session_state.get("dist_key", "normal")
    if current_key not in dist_keys:
        current_key = "normal"

    selected_key = st.sidebar.selectbox(
        t("select_distribution", lang),
        options=dist_keys,
        index=dist_keys.index(current_key),
        format_func=lambda k: DISTRIBUTIONS[k].name_no if lang == "no" else DISTRIBUTIONS[k].name_en,
        key="dist_selector",
    )
    if selected_key != st.session_state.get("dist_key"):
        st.session_state["dist_key"] = selected_key
        st.session_state["ghost_dist"] = None

    spec: DistributionSpec = DISTRIBUTIONS[selected_key]
    parsed = cached_distribution_parsed(selected_key)

    view_choice = st.sidebar.radio(
        t("dist_view_type", lang),
        options=["pdf", "cdf"],
        format_func=lambda v: t("view_pdf", lang) if v == "pdf" else t("view_cdf", lang),
        horizontal=False,
        key="dist_view_radio",
    )

    st.sidebar.markdown(f"### {t('parameters_header', lang)}")

    url_seed: Dict[str, float] = st.session_state.get("url_params_seed", {})
    param_values: Dict[str, float] = {}

    for p in spec.params:
        p_label = p.label_no if lang == "no" else p.label_en
        default_v = url_seed.get(p.name, p.default)
        default_v = max(p.min_val, min(p.max_val, default_v))
        if p.is_int:
            val = st.sidebar.slider(
                p_label,
                min_value=int(p.min_val),
                max_value=int(p.max_val),
                value=int(round(default_v)),
                step=int(p.step),
                key=f"dist_{selected_key}_{p.name}",
            )
            param_values[p.name] = float(val)
        else:
            val = st.sidebar.slider(
                p_label,
                min_value=float(p.min_val),
                max_value=float(p.max_val),
                value=float(default_v),
                step=float(p.step),
                key=f"dist_{selected_key}_{p.name}",
            )
            param_values[p.name] = float(val)

    # Guard Uniform b > a
    if selected_key == "uniform" and param_values["b"] <= param_values["a"] + 0.1:
        param_values["b"] = param_values["a"] + 0.2

    # Focus point slider x0 / k0
    f_min, f_max, f_step = spec.focus_bounds_fn(param_values)
    f_default = url_seed.get(spec.indep_var, spec.default_focus)
    f_default = max(f_min, min(f_max, f_default))
    focus_key = f"dist_{selected_key}_focus"

    if spec.kind == "discrete":
        f_min_i = int(math.floor(f_min))
        f_max_i = max(f_min_i + 1, int(math.ceil(f_max)))
        if focus_key in st.session_state:
            st.session_state[focus_key] = max(
                f_min_i, min(f_max_i, int(round(st.session_state[focus_key])))
            )
        focus_val = float(
            st.sidebar.slider(
                f"{t('focus_point_label', lang)} ({spec.indep_var}₀)",
                min_value=f_min_i,
                max_value=f_max_i,
                value=int(round(f_default)),
                step=1,
                help=t("focus_point_help", lang),
                key=focus_key,
            )
        )
    else:
        if f_max <= f_min:
            f_max = f_min + 1.0
        if focus_key in st.session_state:
            st.session_state[focus_key] = max(
                float(f_min), min(float(f_max), float(st.session_state[focus_key]))
            )
        focus_val = float(
            st.sidebar.slider(
                f"{t('focus_point_label', lang)} ({spec.indep_var}₀)",
                min_value=float(f_min),
                max_value=float(f_max),
                value=float(f_default),
                step=float(f_step),
                help=t("focus_point_help", lang),
                key=focus_key,
            )
        )

    # Clear one-time URL seed after first render
    if "url_params_seed" in st.session_state:
        del st.session_state["url_params_seed"]

    full_values = dict(param_values)
    full_values[spec.indep_var] = focus_val

    # Ghost curve controls in sidebar
    st.sidebar.markdown(f"### {t('ghost_section', lang)}")
    g_col1, g_col2 = st.sidebar.columns(2)
    x_grid, y_grid = evaluate_distribution_curve(spec, param_values, view_type=view_choice)
    stats_now = compute_distribution_stats(spec, param_values)

    if g_col1.button(t("ghost_save", lang), use_container_width=True, key="save_ghost_dist"):
        st.session_state["ghost_dist"] = {
            "dist_key": selected_key,
            "view_type": view_choice,
            "x": x_grid.copy(),
            "y": y_grid.copy(),
            "stats": dict(stats_now),
        }
    if g_col2.button(t("ghost_clear", lang), use_container_width=True, key="clear_ghost_dist"):
        st.session_state["ghost_dist"] = None

    ghost_state = st.session_state.get("ghost_dist")
    if ghost_state and (
        ghost_state["dist_key"] != selected_key or ghost_state["view_type"] != view_choice
    ):
        ghost_state = None

    if ghost_state:
        st.sidebar.caption(t("ghost_active_note", lang))

    # Baseline for smooth response scaling
    base_sig = f"dist::{selected_key}"
    base_defaults = {p.name: float(p.default) for p in spec.params}
    base_defaults[spec.indep_var] = float(spec.default_focus)
    baseline = _get_or_create_baseline(base_sig, parsed, base_defaults)

    if st.sidebar.button(t("reset_baseline", lang), use_container_width=True, key="reset_base_dist"):
        probe = evaluate_formula_point(parsed, full_values, scaling_mode="sensitivity")
        st.session_state["baselines"][base_sig] = {
            "output": probe.output_value if (probe.valid and abs(probe.output_value) > 1e-12) else 1.0,
            "values": dict(full_values),
            "partials": {oid: o.partial_derivative for oid, o in probe.occurrences.items()},
        }
        baseline = st.session_state["baselines"][base_sig]

    # Evaluate formula sensitivity & visual scales at focus point
    eval_res = evaluate_formula_point(
        parsed=parsed,
        values=full_values,
        scaling_mode=scaling_mode,
        baseline_output=baseline["output"],
        baseline_values=baseline["values"],
        baseline_partials=baseline["partials"],
    )

    # --- MAIN STAGE ---
    desc = spec.description_no if lang == "no" else spec.description_en
    dist_title = spec.name_no if lang == "no" else spec.name_en
    st.markdown(f"**{dist_title}** — {desc}")

    # 1. Hero Dynamic Formula Card
    st.markdown(
        render_formula_card_html(
            parsed=parsed,
            eval_res=eval_res,
            scaling_mode=scaling_mode,
            color_coding=color_coding,
            lang=lang,
        ),
        unsafe_allow_html=True,
    )

    # 2. Intuition insight box
    insight = spec.insight_no if lang == "no" else spec.insight_en
    st.markdown(
        f'<div class="fv-insight-box">💡 <strong>{t("sigma_insight_title", lang)}:</strong> {insight}</div>',
        unsafe_allow_html=True,
    )

    # 3. Occurrence breakdown strip
    with st.container():
        st.caption(t("breakdown_title", lang))
        st.markdown(
            render_breakdown_html(eval_res, color_coding=color_coding, lang=lang),
            unsafe_allow_html=True,
        )

    # 4. Key metrics row
    dist_obj = spec.scipy_dist(param_values)
    if view_choice == "cdf":
        focus_y_plot = float(dist_obj.cdf(focus_val))
        y_axis_label = t("chart_cdf", lang)
    else:
        focus_y_plot = (
            float(dist_obj.pmf(focus_val)) if spec.kind == "discrete" else float(dist_obj.pdf(focus_val))
        )
        y_axis_label = t("chart_probability", lang) if spec.kind == "discrete" else t("chart_density", lang)

    ghost_stats = ghost_state["stats"] if ghost_state else {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        f"{t('metric_response', lang)} ({spec.indep_var}₀={focus_val:.3g})",
        _format_metric(focus_y_plot),
    )
    m2.metric(
        t("metric_mean", lang),
        _format_metric(stats_now["mean"]),
        delta=_format_delta(stats_now["mean"], ghost_stats.get("mean")),
    )
    m3.metric(
        t("metric_variance", lang),
        _format_metric(stats_now["var"]),
        delta=_format_delta(stats_now["var"], ghost_stats.get("var")),
    )
    m4.metric(
        t("metric_std", lang),
        _format_metric(stats_now["std"]),
        delta=_format_delta(stats_now["std"], ghost_stats.get("std")),
    )

    # 5. Interactive Plotly Chart
    ghost_xy = (ghost_state["x"], ghost_state["y"]) if ghost_state else None
    fig = build_distribution_chart(
        x_grid=x_grid,
        y_grid=y_grid,
        kind=spec.kind,
        view_type=view_choice,
        x_label=spec.indep_var,
        y_label=y_axis_label,
        title=f"{dist_title} — {'CDF' if view_choice == 'cdf' else ('PMF' if spec.kind == 'discrete' else 'PDF')}",
        focus_x=focus_val,
        focus_y=focus_y_plot,
        mean_val=stats_now["mean"],
        ghost_data=ghost_xy,
        lang=lang,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6. Expandable mathematical breakdown table
    with st.expander(t("details_expander", lang), expanded=False):
        _render_details_table(eval_res, lang)

    return full_values


def _on_example_selected() -> None:
    sel_key = st.session_state.get("example_picker", "")
    if not sel_key:
        return
    for ex in EXAMPLE_FORMULAS:
        if ex.key == sel_key:
            st.session_state["custom_formula_text"] = ex.formula
            st.session_state["active_example_key"] = ex.key
            st.session_state["ghost_custom"] = None
            break


def _run_custom_mode(lang: str, scaling_mode: str, color_coding: bool) -> Dict[str, float]:
    """Render sidebar controls and main canvas for user-defined custom formulas."""
    # Apply any pending formula loaded via button on previous run BEFORE text_input renders
    if "_pending_custom_formula" in st.session_state:
        st.session_state["custom_formula_text"] = st.session_state.pop("_pending_custom_formula")

    # Example library picker
    example_options = [""] + [ex.key for ex in EXAMPLE_FORMULAS]
    ex_titles = {
        ex.key: (ex.title_no if lang == "no" else ex.title_en) for ex in EXAMPLE_FORMULAS
    }
    st.sidebar.selectbox(
        t("example_library", lang),
        options=example_options,
        format_func=lambda k: t("example_choose", lang) if k == "" else ex_titles[k],
        key="example_picker",
        on_change=_on_example_selected,
    )

    # Main formula text input
    raw_formula = st.sidebar.text_input(
        t("custom_formula_input", lang),
        key="custom_formula_text",
        help=t("custom_formula_help", lang),
    )

    try:
        parsed = cached_custom_parsed(raw_formula)
    except ValueError as err:
        st.error(t("error_parse", lang, err=str(err)))
        return {}

    matched_example = _get_example_by_formula(raw_formula)
    if matched_example:
        ex_desc = matched_example.description_no if lang == "no" else matched_example.description_en
        st.markdown(f"**{ex_titles[matched_example.key]}** — {ex_desc}")

    # Parameter sliders
    st.sidebar.markdown(f"### {t('parameters_header', lang)}")
    url_seed: Dict[str, float] = st.session_state.get("url_params_seed", {})
    custom_ranges: Dict[str, Tuple[float, float, float]] = st.session_state["custom_ranges"]

    # Optional slider range customization expander
    with st.sidebar.expander(t("custom_ranges_expander", lang), expanded=False):
        for var in parsed.variables:
            disp = parsed.display_names[var]
            rkey = f"{parsed.raw_formula}::{var}"
            base_def = (
                matched_example.defaults.get(disp, parsed.default_values[var])
                if matched_example
                else parsed.default_values[var]
            )
            cur_min, cur_max, cur_step = custom_ranges.get(
                rkey, _default_range_for_var(var, disp, base_def, matched_example)
            )
            c1, c2 = st.columns(2)
            new_min = c1.number_input(f"{disp} min", value=float(cur_min), key=f"rmin_{rkey}")
            new_max = c2.number_input(f"{disp} max", value=float(cur_max), key=f"rmax_{rkey}")
            if new_max > new_min:
                new_step = round((new_max - new_min) / 100.0, 4) or 0.01
                custom_ranges[rkey] = (float(new_min), float(new_max), float(new_step))

    values: Dict[str, float] = {}
    defaults_map: Dict[str, float] = {}
    for var in parsed.variables:
        disp = parsed.display_names[var]
        rkey = f"{parsed.raw_formula}::{var}"
        base_def = (
            matched_example.defaults.get(disp, parsed.default_values[var])
            if matched_example
            else parsed.default_values[var]
        )
        defaults_map[var] = float(base_def)
        lo, hi, step = custom_ranges.get(
            rkey, _default_range_for_var(var, disp, base_def, matched_example)
        )
        seed_val = url_seed.get(var, url_seed.get(disp, base_def))
        seed_val = max(lo, min(hi, float(seed_val)))
        val_key = f"cval_{rkey}"

        if parsed.output_type == "discrete_func" and var == parsed.independent_var:
            lo_i, hi_i = int(math.floor(lo)), max(int(math.floor(lo)) + 1, int(math.ceil(hi)))
            if val_key in st.session_state:
                st.session_state[val_key] = max(lo_i, min(hi_i, int(round(st.session_state[val_key]))))
            val = float(
                st.sidebar.slider(
                    f"{disp} ({t('focus_point_label', lang)})",
                    min_value=lo_i,
                    max_value=hi_i,
                    value=int(round(seed_val)),
                    step=1,
                    key=val_key,
                )
            )
        else:
            if val_key in st.session_state:
                st.session_state[val_key] = max(float(lo), min(float(hi), float(st.session_state[val_key])))
            label = (
                f"{disp} ({t('focus_point_label', lang)})"
                if var == parsed.independent_var
                else disp
            )
            val = float(
                st.sidebar.slider(
                    label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(seed_val),
                    step=float(step),
                    key=val_key,
                )
            )
        values[var] = val

    if "url_params_seed" in st.session_state:
        del st.session_state["url_params_seed"]

    # Choose sweep variable for 2D plot
    sweep_var: Optional[str] = None
    if parsed.variables:
        default_sweep = (
            parsed.independent_var
            if parsed.independent_var in parsed.variables
            else parsed.variables[-1]
        )
        # For TF-IDF, defaulting x-axis sweep to df(t) or a is extra intuitive
        if matched_example and matched_example.key == "tfidf":
            for v in parsed.variables:
                if "df" in parsed.display_names[v].lower():
                    default_sweep = v
                    break

        sweep_idx = parsed.variables.index(default_sweep) if default_sweep in parsed.variables else 0
        sweep_var = st.sidebar.selectbox(
            t("sweep_var_label", lang),
            options=parsed.variables,
            index=sweep_idx,
            format_func=lambda v: parsed.display_names[v],
            help=t("sweep_var_help", lang),
            key=f"sweep_{parsed.raw_formula}",
        )

    # Baseline tracking
    base_sig = f"custom::{parsed.raw_formula}"
    baseline = _get_or_create_baseline(base_sig, parsed, defaults_map)

    # Compute 2D curve data
    x_grid: Optional[np.ndarray] = None
    y_grid: Optional[np.ndarray] = None
    if sweep_var is not None:
        rkey = f"{parsed.raw_formula}::{sweep_var}"
        lo, hi, _ = custom_ranges.get(
            rkey,
            _default_range_for_var(
                sweep_var, parsed.display_names[sweep_var], defaults_map[sweep_var], matched_example
            ),
        )
        if parsed.output_type == "discrete_func" and sweep_var == parsed.independent_var:
            x_grid = np.arange(int(math.floor(lo)), int(math.ceil(hi)) + 1, dtype=float)
        else:
            x_grid = np.linspace(lo, hi, 260)
        y_grid = evaluate_curve(parsed, sweep_var, x_grid, values)

    # Ghost reference controls
    st.sidebar.markdown(f"### {t('ghost_section', lang)}")
    gc1, gc2 = st.sidebar.columns(2)
    if gc1.button(t("ghost_save", lang), use_container_width=True, key="save_ghost_custom"):
        if x_grid is not None and y_grid is not None:
            st.session_state["ghost_custom"] = {
                "formula": parsed.raw_formula,
                "sweep_var": sweep_var,
                "x": x_grid.copy(),
                "y": y_grid.copy(),
            }
    if gc2.button(t("ghost_clear", lang), use_container_width=True, key="clear_ghost_custom"):
        st.session_state["ghost_custom"] = None

    ghost_c = st.session_state.get("ghost_custom")
    if ghost_c and (ghost_c["formula"] != parsed.raw_formula or ghost_c["sweep_var"] != sweep_var):
        ghost_c = None
    if ghost_c:
        st.sidebar.caption(t("ghost_active_note", lang))

    if st.sidebar.button(t("reset_baseline", lang), use_container_width=True, key="reset_base_custom"):
        probe = evaluate_formula_point(parsed, values, scaling_mode="sensitivity")
        st.session_state["baselines"][base_sig] = {
            "output": probe.output_value if (probe.valid and abs(probe.output_value) > 1e-12) else 1.0,
            "values": dict(values),
            "partials": {oid: o.partial_derivative for oid, o in probe.occurrences.items()},
        }
        baseline = st.session_state["baselines"][base_sig]

    # Evaluate formula at current values
    eval_res = evaluate_formula_point(
        parsed=parsed,
        values=values,
        scaling_mode=scaling_mode,
        baseline_output=baseline["output"],
        baseline_values=baseline["values"],
        baseline_partials=baseline["partials"],
    )

    # Saved formulas panel in sidebar
    st.sidebar.divider()
    st.sidebar.markdown(f"### {t('saved_formulas_header', lang)}")
    save_name = st.sidebar.text_input(
        t("save_name_input", lang),
        value=parsed.lhs_display,
        key="save_formula_name_input",
    )
    if st.sidebar.button(t("save_button", lang), use_container_width=True, key="btn_save_formula"):
        entry = {
            "name": save_name.strip() or parsed.lhs_display,
            "formula": parsed.raw_formula,
            "params": {k: float(v) for k, v in values.items()},
        }
        saved_list: List[Dict[str, Any]] = st.session_state["saved_formulas"]
        # Update existing name or append
        saved_list = [item for item in saved_list if item["name"] != entry["name"]]
        saved_list.append(entry)
        st.session_state["saved_formulas"] = saved_list
        st.sidebar.success(t("saved_confirmation", lang, name=entry["name"]))

    saved_list = st.session_state["saved_formulas"]
    if saved_list:
        names = [item["name"] for item in saved_list]
        chosen_saved = st.sidebar.selectbox(
            t("load_saved_label", lang),
            options=names,
            key="saved_formula_select",
        )
        sc1, sc2 = st.sidebar.columns(2)
        if sc1.button(f"↩ {t('load_saved_button', lang)}", use_container_width=True, key="btn_load_saved"):
            for item in saved_list:
                if item["name"] == chosen_saved:
                    st.session_state["_pending_custom_formula"] = item["formula"]
                    st.session_state["url_params_seed"] = dict(item.get("params", {}))
                    st.rerun()
        if sc2.button(t("delete_saved_button", lang), use_container_width=True, key="btn_del_saved"):
            st.session_state["saved_formulas"] = [
                item for item in saved_list if item["name"] != chosen_saved
            ]
            st.rerun()
    else:
        st.sidebar.caption(t("no_saved_yet", lang))

    # --- MAIN STAGE ---
    # 1. Hero Dynamic Formula Card
    st.markdown(
        render_formula_card_html(
            parsed=parsed,
            eval_res=eval_res,
            scaling_mode=scaling_mode,
            color_coding=color_coding,
            lang=lang,
        ),
        unsafe_allow_html=True,
    )

    if not eval_res.valid:
        st.warning(t("error_eval", lang))

    # 2. Occurrence breakdown strip
    st.caption(t("breakdown_title", lang))
    st.markdown(
        render_breakdown_html(eval_res, color_coding=color_coding, lang=lang),
        unsafe_allow_html=True,
    )

    # 3. Output-type & response metrics strip
    type_label_map = {
        "scalar": t("output_scalar", lang),
        "continuous_func": t("output_continuous_func", lang),
        "discrete_func": t("output_discrete_func", lang),
        "continuous_dist": t("output_continuous_dist", lang),
        "discrete_dist": t("output_discrete_dist", lang),
    }
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        f"{t('metric_scalar_response', lang)} ({parsed.lhs_display})",
        _format_metric(eval_res.output_value) if eval_res.valid else "—",
    )
    mc2.metric(
        t("metric_output_type", lang),
        type_label_map.get(parsed.output_type, parsed.output_type),
    )
    mc3.metric(
        t("metric_dominant", lang),
        _find_dominant_occurrence(eval_res, lang),
    )

    # 4. Interactive 2D Plot (both for functions AND for scalar 2D sweep)
    if sweep_var is not None and x_grid is not None and y_grid is not None:
        if parsed.output_type == "scalar":
            st.info(
                t(
                    "chart_scalar_note",
                    lang,
                    lhs=parsed.lhs_display,
                    val=_format_metric(eval_res.output_value),
                    xvar=parsed.display_names[sweep_var],
                )
            )
        if np.any(~np.isfinite(y_grid)):
            st.caption(t("domain_warning", lang))

        ghost_xy = (ghost_c["x"], ghost_c["y"]) if ghost_c else None
        fig = build_custom_curve_chart(
            x_grid=x_grid,
            y_grid=y_grid,
            is_discrete=(parsed.output_type == "discrete_func" and sweep_var == parsed.independent_var),
            x_label=parsed.display_names[sweep_var],
            y_label=parsed.lhs_display,
            current_x=values[sweep_var],
            current_y=eval_res.output_value if eval_res.valid else float("nan"),
            ghost_data=ghost_xy,
            lang=lang,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5. Detailed mathematical breakdown table
    with st.expander(t("details_expander", lang), expanded=False):
        _render_details_table(eval_res, lang)

    return values


def main() -> None:
    _init_session_state()

    # Header + Language Switcher
    h_col1, h_col2 = st.columns([4, 1])
    with h_col2:
        lang_choice = st.radio(
            t("language_label", st.session_state["lang"]),
            options=["no", "en"],
            index=0 if st.session_state["lang"] == "no" else 1,
            format_func=lambda code: "🇳🇴 Norsk" if code == "no" else "🇬🇧 English",
            horizontal=True,
            key="lang_toggle",
        )
        st.session_state["lang"] = lang_choice

    lang = st.session_state["lang"]

    with h_col1:
        st.title(f"📐 {t('app_title', lang)}")
        st.caption(t("app_subtitle", lang))

    # Sidebar: Mode selection
    mode = st.sidebar.radio(
        t("sidebar_mode", lang),
        options=["distributions", "custom"],
        index=0 if st.session_state["mode"] == "distributions" else 1,
        format_func=lambda m: (
            f"📊 {t('mode_distributions', lang)}"
            if m == "distributions"
            else f"✍️ {t('mode_custom', lang)}"
        ),
        key="mode_radio",
    )
    st.session_state["mode"] = mode

    # Sidebar: Visual Formula Scaling controls
    st.sidebar.divider()
    st.sidebar.markdown(f"### {t('scaling_section', lang)}")
    scaling_mode = st.sidebar.radio(
        t("scaling_mode_label", lang),
        options=["sensitivity", "value"],
        format_func=lambda s: (
            f"⚡ {t('scaling_sensitivity', lang)}"
            if s == "sensitivity"
            else f"📏 {t('scaling_value', lang)}"
        ),
        help=t("scaling_mode_help", lang),
        key="scaling_mode_radio",
    )
    color_coding = st.sidebar.checkbox(
        t("color_coding_label", lang),
        value=True,
        key="color_coding_check",
    )
    st.sidebar.divider()

    # Run active mode
    if mode == "distributions":
        active_params = _run_distributions_mode(lang, scaling_mode, color_coding)
    else:
        active_params = _run_custom_mode(lang, scaling_mode, color_coding)

    # Share / Export / Import section
    with st.expander(f"🔗 {t('export_import_expander', lang)}", expanded=False):
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.write(f"**{t('share_link_label', lang)}**")
            if st.button(t("share_link_button", lang), key="btn_share_url"):
                st.query_params["lang"] = lang
                st.query_params["mode"] = mode
                if mode == "distributions":
                    st.query_params["dist"] = st.session_state["dist_key"]
                    if "formula" in st.query_params:
                        del st.query_params["formula"]
                else:
                    st.query_params["formula"] = st.session_state["custom_formula_text"]
                    if "dist" in st.query_params:
                        del st.query_params["dist"]
                st.query_params["params"] = json.dumps(active_params)
                st.success(t("share_link_copied", lang))

        with s_col2:
            saved_json = json.dumps(st.session_state.get("saved_formulas", []), indent=2, ensure_ascii=False)
            st.download_button(
                label=t("export_json_button", lang),
                data=saved_json,
                file_name="saved_formulas.json",
                mime="application/json",
                key="btn_export_json",
            )
            uploaded = st.file_uploader(
                t("import_json_label", lang),
                type=["json"],
                key="upload_saved_json",
            )
            if uploaded is not None:
                try:
                    data = json.loads(uploaded.read().decode("utf-8"))
                    if isinstance(data, list):
                        valid_items = [
                            item
                            for item in data
                            if isinstance(item, dict) and "formula" in item and "name" in item
                        ]
                        existing = st.session_state.get("saved_formulas", [])
                        existing_names = {x["name"] for x in existing}
                        added = 0
                        for item in valid_items:
                            if item["name"] not in existing_names:
                                existing.append(item)
                                existing_names.add(item["name"])
                                added += 1
                        st.session_state["saved_formulas"] = existing
                        st.success(t("import_success", lang, count=added))
                    else:
                        st.error(t("import_error", lang))
                except Exception:
                    st.error(t("import_error", lang))


if __name__ == "__main__":
    main()
