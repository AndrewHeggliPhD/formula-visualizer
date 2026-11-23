"""FormelVis / FormulaViz — Interactive 60 FPS Visual Explorer for Statistical Distributions & Formulas."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from src.distributions import DISTRIBUTIONS, DistributionSpec, get_distribution_parsed
from src.examples import EXAMPLE_FORMULAS, FormulaExample
from src.formula_engine import (
    ParsedFormula,
    evaluate_formula_point,
    parse_custom_formula,
)
from src.i18n import t
from src.live_widget import render_live_studio_html

st.set_page_config(
    page_title="FormelVis / FormulaViz",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_theme_css(dark_mode: bool) -> None:
    """Apply crisp Light or Dark theme styling across Streamlit chrome & widgets."""
    if dark_mode:
        css = """
        <style>
        .block-container { padding-top: 0.75rem !important; padding-bottom: 1rem !important; max-width: 1440px; }
        header[data-testid="stHeader"] { background: #0B1120 !important; }
        [data-testid="stAppViewContainer"] { background: #0B1120 !important; color: #F1F5F9 !important; }
        [data-testid="stSidebar"] { background: #111827 !important; border-right: 1px solid #1E293B !important; }
        [data-testid="stSidebar"] * { color: #E2E8F0; }
        h1, h2, h3, p, span, label, div[data-testid="stMarkdownContainer"] { color: #F1F5F9; }
        .stCaption, [data-testid="stCaptionContainer"] { color: #94A3B8 !important; }
        input, textarea, select, [data-baseweb="select"] > div, [data-baseweb="input"] > div {
            background-color: #1E293B !important;
            color: #F1F5F9 !important;
            border-color: #334155 !important;
        }
        [data-baseweb="popover"] ul, [data-baseweb="menu"] {
            background-color: #1E293B !important;
            color: #F1F5F9 !important;
        }
        button[kind="secondary"] {
            background-color: #1E293B !important;
            color: #F1F5F9 !important;
            border-color: #334155 !important;
        }
        [data-testid="stExpander"] {
            background-color: #111827 !important;
            border-color: #334155 !important;
        }
        iframe { border: none !important; background: transparent !important; }
        </style>
        """
    else:
        css = """
        <style>
        .block-container { padding-top: 0.75rem !important; padding-bottom: 1rem !important; max-width: 1440px; }
        header[data-testid="stHeader"] { background: #F8FAFC !important; }
        [data-testid="stAppViewContainer"] { background: #F8FAFC !important; color: #0F172A !important; }
        [data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; }
        iframe { border: none !important; background: transparent !important; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


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

    theme = qp.get("theme", "light")
    if theme not in ("light", "dark"):
        theme = "light"
    st.session_state["theme"] = theme

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

    st.session_state["saved_formulas"] = []
    st.session_state["custom_ranges"] = {}

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


def _on_example_selected() -> None:
    sel_key = st.session_state.get("example_picker", "")
    if not sel_key:
        return
    for ex in EXAMPLE_FORMULAS:
        if ex.key == sel_key:
            st.session_state["custom_formula_text"] = ex.formula
            break


def _run_distributions_mode(lang: str, dark_mode: bool) -> Dict[str, float]:
    """Configure a built-in distribution and render the 60 FPS real-time studio."""
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
    st.session_state["dist_key"] = selected_key

    spec: DistributionSpec = DISTRIBUTIONS[selected_key]
    parsed = cached_distribution_parsed(selected_key)
    url_seed: Dict[str, float] = st.session_state.pop("url_params_seed", {})

    sliders: List[Dict[str, Any]] = []
    init_values: Dict[str, float] = {}

    for p in spec.params:
        p_label = p.label_no if lang == "no" else p.label_en
        val = max(p.min_val, min(p.max_val, float(url_seed.get(p.name, p.default))))
        if p.is_int:
            val = float(int(round(val)))
        init_values[p.name] = val
        sliders.append(
            {
                "name": p.name,
                "label": p_label,
                "min": float(p.min_val),
                "max": float(p.max_val),
                "step": float(p.step),
                "value": val,
                "is_int": p.is_int,
                "is_focus": False,
            }
        )

    f_min, f_max, f_step = spec.focus_bounds_fn(init_values)
    if selected_key == "normal":
        f_min, f_max = -6.0, 6.0
    elif selected_key == "uniform":
        f_min, f_max = -5.0, 8.0
    f_val = max(f_min, min(f_max, float(url_seed.get(spec.indep_var, spec.default_focus))))
    if spec.kind == "discrete":
        f_val = float(int(round(f_val)))
    init_values[spec.indep_var] = f_val

    sliders.append(
        {
            "name": spec.indep_var,
            "label": f"{t('focus_point_label', lang)} ({spec.indep_var}₀)",
            "min": float(f_min),
            "max": float(f_max),
            "step": 1.0 if spec.kind == "discrete" else float(f_step),
            "value": f_val,
            "is_int": spec.kind == "discrete",
            "is_focus": True,
        }
    )

    initial_eval = evaluate_formula_point(parsed, init_values, scaling_mode="sensitivity")

    dist_title = spec.name_no if lang == "no" else spec.name_en
    insight = spec.insight_no if lang == "no" else spec.insight_en

    studio_html = render_live_studio_html(
        parsed=parsed,
        initial_eval=initial_eval,
        sliders=sliders,
        mode_kind=parsed.output_type,
        dist_key=selected_key,
        sweep_var=spec.indep_var,
        sweep_options=[{"name": spec.indep_var, "label": spec.indep_var}],
        title=dist_title,
        insight_text=insight,
        lang=lang,
        dark_mode=dark_mode,
        initial_scaling_mode="sensitivity",
        initial_color_coding=True,
        initial_view_type="pdf",
    )
    components.html(studio_html, height=645, scrolling=True)
    return init_values


def _run_custom_mode(lang: str, dark_mode: bool) -> Dict[str, float]:
    """Configure a user-defined custom formula and render the 60 FPS real-time studio."""
    if "_pending_custom_formula" in st.session_state:
        st.session_state["custom_formula_text"] = st.session_state.pop("_pending_custom_formula")

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
    url_seed: Dict[str, float] = st.session_state.pop("url_params_seed", {})
    custom_ranges: Dict[str, Tuple[float, float, float]] = st.session_state["custom_ranges"]

    with st.sidebar.expander(t("custom_ranges_expander", lang), expanded=False):
        for var in parsed.variables:
            disp = parsed.display_names[var]
            rkey = f"{parsed.raw_formula}::{var}"
            base_def = (
                matched_example.defaults.get(disp, parsed.default_values[var])
                if matched_example
                else parsed.default_values[var]
            )
            cur_min, cur_max, _ = custom_ranges.get(
                rkey, _default_range_for_var(var, disp, base_def, matched_example)
            )
            c1, c2 = st.columns(2)
            new_min = c1.number_input(f"{disp} min", value=float(cur_min), key=f"rmin_{rkey}")
            new_max = c2.number_input(f"{disp} max", value=float(cur_max), key=f"rmax_{rkey}")
            if new_max > new_min:
                new_step = round((new_max - new_min) / 100.0, 4) or 0.01
                custom_ranges[rkey] = (float(new_min), float(new_max), float(new_step))

    sliders: List[Dict[str, Any]] = []
    init_values: Dict[str, float] = {}
    sweep_options: List[Dict[str, str]] = []

    for var in parsed.variables:
        disp = parsed.display_names[var]
        rkey = f"{parsed.raw_formula}::{var}"
        base_def = (
            matched_example.defaults.get(disp, parsed.default_values[var])
            if matched_example
            else parsed.default_values[var]
        )
        lo, hi, step = custom_ranges.get(
            rkey, _default_range_for_var(var, disp, base_def, matched_example)
        )
        seed_val = max(lo, min(hi, float(url_seed.get(var, url_seed.get(disp, base_def)))))
        is_disc_indep = parsed.output_type == "discrete_func" and var == parsed.independent_var
        if is_disc_indep:
            seed_val = float(int(round(seed_val)))

        init_values[var] = seed_val
        sweep_options.append({"name": var, "label": disp})
        sliders.append(
            {
                "name": var,
                "label": f"{disp} ({t('focus_point_label', lang)})" if var == parsed.independent_var else disp,
                "min": float(lo),
                "max": float(hi),
                "step": 1.0 if is_disc_indep else float(step),
                "value": seed_val,
                "is_int": is_disc_indep,
                "is_focus": (var == parsed.independent_var),
            }
        )

    default_sweep: Optional[str] = None
    if parsed.variables:
        default_sweep = (
            parsed.independent_var
            if parsed.independent_var in parsed.variables
            else parsed.variables[-1]
        )
        if matched_example and matched_example.key == "tfidf":
            for v in parsed.variables:
                if "df" in parsed.display_names[v].lower():
                    default_sweep = v
                    break

    # Saved formulas panel in sidebar
    st.sidebar.divider()
    st.sidebar.markdown(f"**{t('saved_formulas_header', lang)}**")
    save_name = st.sidebar.text_input(
        t("save_name_input", lang),
        value=parsed.lhs_display,
        key="save_formula_name_input",
    )
    if st.sidebar.button(t("save_button", lang), use_container_width=True, key="btn_save_formula"):
        entry = {
            "name": save_name.strip() or parsed.lhs_display,
            "formula": parsed.raw_formula,
            "params": {k: float(v) for k, v in init_values.items()},
        }
        saved_list: List[Dict[str, Any]] = [
            item for item in st.session_state["saved_formulas"] if item["name"] != entry["name"]
        ]
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

    initial_eval = evaluate_formula_point(parsed, init_values, scaling_mode="sensitivity")
    title = ex_titles[matched_example.key] if matched_example else parsed.lhs_display
    insight = (
        (matched_example.description_no if lang == "no" else matched_example.description_en)
        if matched_example
        else None
    )

    studio_html = render_live_studio_html(
        parsed=parsed,
        initial_eval=initial_eval,
        sliders=sliders,
        mode_kind=parsed.output_type,
        dist_key=None,
        sweep_var=default_sweep,
        sweep_options=sweep_options,
        title=title,
        insight_text=insight,
        lang=lang,
        dark_mode=dark_mode,
        initial_scaling_mode="sensitivity",
        initial_color_coding=True,
        initial_view_type="pdf",
    )
    components.html(studio_html, height=645, scrolling=True)
    return init_values


def main() -> None:
    _init_session_state()

    # Compact single-row header with Theme (Dark/Light) & Language toggles
    h_col1, h_col2, h_col3 = st.columns([5, 1.7, 1.7])
    cur_lang = st.session_state["lang"]
    theme_labels = {
        "light": f"☀️ {t('theme_light', cur_lang)}",
        "dark": f"🌙 {t('theme_dark', cur_lang)}",
    }

    with h_col2:
        theme_choice = st.radio(
            t("theme_label", cur_lang),
            options=["light", "dark"],
            index=0 if st.session_state["theme"] == "light" else 1,
            format_func=lambda th: theme_labels.get(th, th),
            horizontal=True,
            label_visibility="collapsed",
            key="theme_toggle",
        )
        st.session_state["theme"] = theme_choice

    with h_col3:
        lang_choice = st.radio(
            t("language_label", cur_lang),
            options=["no", "en"],
            index=0 if cur_lang == "no" else 1,
            format_func=lambda code: "🇳🇴 NO" if code == "no" else "🇬🇧 EN",
            horizontal=True,
            label_visibility="collapsed",
            key="lang_toggle",
        )
        st.session_state["lang"] = lang_choice

    lang = st.session_state["lang"]
    dark_mode = st.session_state["theme"] == "dark"
    _inject_theme_css(dark_mode)

    with h_col1:
        st.markdown(
            f"<div style='display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;'>"
            f"<span style='font-size:1.45rem;font-weight:800;'>📐 {t('app_title', lang)}</span>"
            f"<span style='font-size:0.86rem;opacity:0.72;'>{t('app_subtitle', lang)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

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

    # Run active mode (renders 60 FPS studio with formula + chart side-by-side above the fold!)
    if mode == "distributions":
        active_params = _run_distributions_mode(lang, dark_mode)
    else:
        active_params = _run_custom_mode(lang, dark_mode)

    # Share / Export / Import in sidebar bottom so main stage stays zero-scroll!
    st.sidebar.divider()
    with st.sidebar.expander(f"🔗 {t('export_import_expander', lang)}", expanded=False):
        if st.button(t("share_link_button", lang), use_container_width=True, key="btn_share_url"):
            st.query_params["lang"] = lang
            st.query_params["theme"] = st.session_state["theme"]
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

        saved_json = json.dumps(
            st.session_state.get("saved_formulas", []), indent=2, ensure_ascii=False
        )
        st.download_button(
            label=t("export_json_button", lang),
            data=saved_json,
            file_name="saved_formulas.json",
            mime="application/json",
            use_container_width=True,
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
