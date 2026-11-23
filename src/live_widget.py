"""60 FPS client-side real-time interactive formula & chart studio (zero server round-trip lag)."""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

from src.formula_engine import EvaluationResult, ParsedFormula
from src.formula_renderer import render_formula_stage_inner_html
from src.i18n import t
from src.js_codegen import compile_parsed_to_js_spec


def _safe_json_for_script(obj: Any) -> str:
    """Serialize to JSON safe for embedding directly inside a <script> tag."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def render_live_studio_html(
    parsed: ParsedFormula,
    initial_eval: EvaluationResult,
    sliders: List[Dict[str, Any]],
    mode_kind: str,
    dist_key: Optional[str],
    sweep_var: Optional[str],
    sweep_options: List[Dict[str, str]],
    title: str,
    insight_text: Optional[str] = None,
    lang: str = "no",
    dark_mode: bool = False,
    initial_scaling_mode: str = "sensitivity",
    initial_color_coding: bool = True,
    initial_view_type: str = "pdf",
) -> str:
    """
    Build a self-contained HTML/CSS/JS interactive studio where sliders, chart scrubbing,
    formula visual scaling, Ghost reference curves, and live metrics all update at 60 FPS
    purely in the browser with zero Streamlit rerun lag.
    """
    js_spec = compile_parsed_to_js_spec(parsed)
    stage_inner_html = render_formula_stage_inner_html(
        parsed=parsed,
        eval_res=initial_eval,
        color_coding=initial_color_coding,
        lang=lang,
        base_rem=1.32,
    )

    occ_list = []
    for oid in parsed.occ_ids_ordered:
        meta = parsed.occurrences[oid]
        occ_list.append(
            {
                "occ_id": oid,
                "var_name": meta.var_name,
                "display_name": meta.display_name,
                "occurrence_index": meta.occurrence_index,
                "total_occurrences": meta.total_occurrences,
                "context": meta.context_no if lang == "no" else meta.context_en,
                "is_independent": meta.is_independent,
            }
        )

    type_label_map = {
        "scalar": t("output_scalar", lang),
        "continuous_func": t("output_continuous_func", lang),
        "discrete_func": t("output_discrete_func", lang),
        "continuous_dist": t("output_continuous_dist", lang),
        "discrete_dist": t("output_discrete_dist", lang),
    }

    config = {
        "mode_kind": mode_kind,
        "dist_key": dist_key,
        "lhs_display": parsed.lhs_display,
        "variables": parsed.variables,
        "display_names": parsed.display_names,
        "independent_var": parsed.independent_var,
        "sweep_var": sweep_var,
        "sweep_options": sweep_options,
        "sliders": sliders,
        "occurrences": occ_list,
        "eval_js": js_spec["eval_js"],
        "tagged_eval_js": js_spec["tagged_eval_js"],
        "elasticity_js": js_spec["elasticity_js"],
        "partial_js": js_spec["partial_js"],
        "scaling_mode": initial_scaling_mode,
        "color_coding": initial_color_coding,
        "view_type": initial_view_type,
        "dark_mode": dark_mode,
        "lang": lang,
        "title": title,
        "output_type_label": type_label_map.get(mode_kind, mode_kind),
        "strings": {
            "scaling_sens": t("scaling_sensitivity", lang),
            "scaling_val": t("scaling_value", lang),
            "hint_sens": t("formula_card_hint_sens", lang),
            "hint_val": t("formula_card_hint_val", lang),
            "legend_pos": t("legend_pos", lang),
            "legend_neg": t("legend_neg", lang),
            "legend_neutral": t("legend_neutral", lang),
            "legend_response": t("legend_response", lang),
            "col_value": t("col_value", lang),
            "col_partial": t("col_partial", lang),
            "col_elasticity": t("col_elasticity", lang),
            "col_direction": t("col_direction", lang),
            "col_scale": t("col_scale", lang),
            "dir_up": t("dir_up", lang),
            "dir_down": t("dir_down", lang),
            "dir_neutral": t("dir_neutral", lang),
            "chart_current": t("chart_current", lang),
            "chart_ghost": t("chart_ghost", lang),
            "chart_focus": t("chart_focus", lang),
            "chart_mean": t("chart_mean", lang),
            "chart_density": t("chart_density", lang),
            "chart_probability": t("chart_probability", lang),
            "chart_cdf": t("chart_cdf", lang),
            "metric_response": t("metric_response", lang),
            "metric_scalar": t("metric_scalar_response", lang),
            "metric_mean": t("metric_mean", lang),
            "metric_var": t("metric_variance", lang),
            "metric_std": t("metric_std", lang),
            "metric_type": t("metric_output_type", lang),
            "metric_dom": t("metric_dominant", lang),
            "ghost_active": t("ghost_active_note", lang),
            "occ_tpl": t("occurrence_label", lang, idx="{idx}"),
            "drag_hint": t("drag_chart_hint", lang),
        },
    }

    theme_class = "dark" if dark_mode else "light"
    insight_html = ""
    if insight_text:
        insight_html = (
            f'<div class="insight-pill">💡 {html.escape(insight_text)}</div>'
        )

    is_dist = mode_kind in ("continuous_dist", "discrete_dist")
    dist_view_html = ""
    if is_dist:
        pdf_act = "active" if initial_view_type == "pdf" else ""
        cdf_act = "active" if initial_view_type == "cdf" else ""
        pdf_lbl = "PMF" if mode_kind == "discrete_dist" else "PDF"
        dist_view_html = f"""
        <div class="ctrl-block">
          <div class="ctrl-label">{html.escape(t('dist_view_type', lang))}</div>
          <div class="seg-group" id="viewTypeSeg">
            <button type="button" class="seg-btn {pdf_act}" data-val="pdf">{pdf_lbl}</button>
            <button type="button" class="seg-btn {cdf_act}" data-val="cdf">CDF</button>
          </div>
        </div>
        """

    sweep_select_html = ""
    if not is_dist and len(sweep_options) > 1:
        opts = "".join(
            f'<option value="{html.escape(o["name"], quote=True)}" '
            f'{"selected" if o["name"] == sweep_var else ""}>'
            f'{html.escape(o["label"])}</option>'
            for o in sweep_options
        )
        sweep_select_html = f"""
        <div class="ctrl-block">
          <label class="ctrl-label" for="sweepSelect">{html.escape(t('sweep_var_label', lang))}</label>
          <select id="sweepSelect" class="studio-select">{opts}</select>
        </div>
        """

    sens_act = "active" if initial_scaling_mode == "sensitivity" else ""
    val_act = "active" if initial_scaling_mode == "value" else ""
    color_checked = "checked" if initial_color_coding else ""

    config_json = _safe_json_for_script(config)

    return f"""<!DOCTYPE html>
<html lang="{lang}" class="{theme_class}">
<head>
<meta charset="utf-8" />
<style>
:root {{
  --bg: #F8FAFC;
  --surface: #FFFFFF;
  --surface-2: #F1F5F9;
  --border: #E2E8F0;
  --text: #0F172A;
  --text-muted: #64748B;
  --accent: #2563EB;
  --accent-soft: rgba(37, 99, 235, 0.12);
  --pos: #059669;
  --pos-bg: rgba(16, 185, 129, 0.14);
  --neg: #DC2626;
  --neg-bg: rgba(239, 68, 68, 0.13);
  --neu: #475569;
  --neu-bg: rgba(148, 163, 184, 0.16);
  --focus: #F59E0B;
  --mean: #7C3AED;
  --grid: #E2E8F0;
  --shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
  --frac-bar: #334155;
  --insight-bg: #EFF6FF;
  --insight-text: #1E3A8A;
}}
html.dark {{
  --bg: #0B1120;
  --surface: #111827;
  --surface-2: #1E293B;
  --border: #334155;
  --text: #F1F5F9;
  --text-muted: #94A3B8;
  --accent: #60A5FA;
  --accent-soft: rgba(96, 165, 250, 0.18);
  --pos: #34D399;
  --pos-bg: rgba(52, 211, 153, 0.18);
  --neg: #F87171;
  --neg-bg: rgba(248, 113, 113, 0.18);
  --neu: #CBD5E1;
  --neu-bg: rgba(148, 163, 184, 0.18);
  --focus: #FBBF24;
  --mean: #A78BFA;
  --grid: #1E293B;
  --shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
  --frac-bar: #CBD5E1;
  --insight-bg: rgba(37, 99, 235, 0.16);
  --insight-text: #BFDBFE;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: transparent;
  color: var(--text);
  overflow-x: hidden;
}}
.studio {{
  display: grid;
  grid-template-columns: 278px minmax(0, 1fr);
  gap: 12px;
  align-items: stretch;
  width: 100%;
}}
@media (max-width: 760px) {{
  .studio {{ grid-template-columns: 1fr; }}
}}
.panel {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
}}
.controls-panel {{
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 615px;
  overflow-y: auto;
}}
.controls-title {{
  font-size: 0.76rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.live-badge {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.68rem;
  font-weight: 700;
  color: var(--pos);
  background: var(--pos-bg);
  padding: 2px 7px;
  border-radius: 999px;
  letter-spacing: 0.02em;
}}
.live-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--pos);
}}
.slider-row {{
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px 8px;
  border-radius: 10px;
  background: var(--surface-2);
  border: 1px solid transparent;
}}
.slider-row.focus-row {{
  border-color: rgba(245, 158, 11, 0.45);
  background: rgba(245, 158, 11, 0.08);
}}
.slider-head {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
}}
.slider-label {{
  font-size: 0.80rem;
  font-weight: 600;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.slider-num {{
  width: 68px;
  padding: 2px 6px;
  font-size: 0.78rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  text-align: right;
  color: var(--text);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
}}
input[type="range"] {{
  width: 100%;
  height: 6px;
  accent-color: var(--accent);
  cursor: pointer;
}}
.slider-row.focus-row input[type="range"] {{
  accent-color: var(--focus);
}}
.ctrl-block {{
  display: flex;
  flex-direction: column;
  gap: 5px;
}}
.ctrl-label {{
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
}}
.seg-group {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
  background: var(--surface-2);
  padding: 3px;
  border-radius: 9px;
  border: 1px solid var(--border);
}}
.seg-btn {{
  border: none;
  background: transparent;
  color: var(--text-muted);
  font-size: 0.75rem;
  font-weight: 600;
  padding: 5px 6px;
  border-radius: 7px;
  cursor: pointer;
  transition: all 0.14s ease;
}}
.seg-btn.active {{
  background: var(--accent);
  color: #FFFFFF;
}}
.check-row {{
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 0.78rem;
  color: var(--text);
  cursor: pointer;
  user-select: none;
}}
.studio-select {{
  width: 100%;
  padding: 5px 8px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text);
  font-size: 0.80rem;
}}
.btn-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}}
.act-btn {{
  padding: 6px 8px;
  font-size: 0.75rem;
  font-weight: 600;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface-2);
  color: var(--text);
  cursor: pointer;
  transition: background 0.14s ease, border-color 0.14s ease;
}}
.act-btn:hover {{
  border-color: var(--accent);
  background: var(--accent-soft);
}}
.act-btn.full {{
  width: 100%;
}}
.ghost-badge {{
  font-size: 0.72rem;
  color: var(--accent);
  display: none;
}}
.ghost-badge.show {{ display: block; }}

/* RIGHT STAGE: FORMULA + CHART + METRICS SIMULTANEOUSLY */
.stage-col {{
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}}
.formula-panel {{
  padding: 10px 16px 10px 16px;
  overflow-x: auto;
}}
.formula-top-bar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 4px;
}}
.formula-title {{
  font-size: 0.73rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
}}
.formula-hint {{
  font-size: 0.72rem;
  color: var(--text-muted);
}}
.fv-math-stage {{
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.35rem;
  min-height: 64px;
  padding: 4px 4px;
  font-family: 'Cambria Math', 'STIX Two Math', 'Latin Modern Math', Georgia, serif;
  font-size: 1.32rem;
  color: var(--text);
  line-height: 1.25;
}}
.fv-eq-sign {{ font-size: 1.4rem; color: var(--text-muted); margin: 0 0.28rem; }}
.fv-rhs-value {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.86rem;
  color: var(--text);
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 2px 10px;
  margin-left: 0.35rem;
  white-space: nowrap;
}}
.fv-tok {{
  display: inline-flex;
  align-items: baseline;
  font-style: italic;
  font-weight: 600;
  padding: 1px 5px;
  border-radius: 7px;
  cursor: help;
  line-height: 1.08;
  white-space: nowrap;
  transition: color 0.1s ease, background-color 0.1s ease;
}}
.fv-tok-pos {{ color: var(--pos); background: var(--pos-bg); border-bottom: 2px solid var(--pos); }}
.fv-tok-neg {{ color: var(--neg); background: var(--neg-bg); border-bottom: 2px solid var(--neg); }}
.fv-tok-neutral {{ color: var(--neu); background: var(--neu-bg); border-bottom: 2px solid var(--neu); }}
.fv-tok-mono {{ color: var(--accent); background: var(--accent-soft); border-bottom: 2px solid var(--accent); }}
.fv-tok-response {{
  color: var(--accent);
  background: var(--accent-soft);
  border: 1.5px solid var(--accent);
  font-style: normal;
  font-weight: 700;
  padding: 2px 9px;
}}
.fv-occ-tag {{ font-style: normal; font-size: 0.54em; vertical-align: sub; opacity: 0.78; margin-left: 1px; }}
.fv-frac {{ display: inline-flex; flex-direction: column; align-items: center; vertical-align: middle; margin: 0 0.14rem; }}
.fv-frac-num {{ display: inline-flex; align-items: center; justify-content: center; padding: 0 0.24rem 0.09rem 0.24rem; border-bottom: 1.8px solid var(--frac-bar); min-width: 1rem; }}
.fv-frac-den {{ display: inline-flex; align-items: center; justify-content: center; padding: 0.12rem 0.24rem 0 0.24rem; min-width: 1rem; }}
.fv-pow {{ display: inline-flex; align-items: flex-start; }}
.fv-sup {{ font-size: 0.72em; margin-left: 1px; margin-top: -0.4em; display: inline-flex; align-items: center; }}
.fv-sqrt {{ display: inline-flex; align-items: center; margin: 0 0.06rem; }}
.fv-sqrt-rad {{ font-size: 1.15em; margin-right: 1px; color: var(--frac-bar); }}
.fv-sqrt-body {{ border-top: 1.5px solid var(--frac-bar); padding: 0.05rem 0.18rem 0 0.14rem; display: inline-flex; align-items: center; }}
.fv-binom {{ display: inline-flex; align-items: center; margin: 0 0.12rem; }}
.fv-binom-paren {{ font-size: 1.65em; font-weight: 300; color: var(--text-muted); line-height: 0.9; }}
.fv-binom-stack {{ display: inline-flex; flex-direction: column; align-items: center; font-size: 0.82em; padding: 0 0.12rem; line-height: 1.08; }}
.fv-fn {{ font-style: normal; font-weight: 500; color: var(--text); margin-right: 1px; }}
.fv-op {{ margin: 0 0.16rem; color: var(--text-muted); font-style: normal; }}
.fv-paren {{ color: var(--text-muted); font-weight: 300; font-size: 1.08em; }}
.fv-num {{ font-style: normal; color: var(--text); }}

.occ-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: center;
  margin-top: 4px;
  padding-top: 6px;
  border-top: 1px dashed var(--border);
}}
.occ-chip {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.72rem;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  color: var(--text);
}}
.occ-chip-dot {{ width: 7px; height: 7px; border-radius: 50%; }}

/* CHART PANEL */
.chart-panel {{
  padding: 10px 14px 8px 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
.chart-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}}
.chart-title {{
  font-size: 0.84rem;
  font-weight: 700;
  color: var(--text);
}}
.chart-hint {{
  font-size: 0.72rem;
  color: var(--text-muted);
}}
.canvas-wrap {{
  position: relative;
  width: 100%;
  height: 290px;
}}
canvas#liveChart {{
  width: 100%;
  height: 100%;
  display: block;
  cursor: crosshair;
  touch-action: none;
}}
.chart-tooltip {{
  position: absolute;
  pointer-events: none;
  display: none;
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 4px 8px;
  font-size: 0.74rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  box-shadow: var(--shadow);
  z-index: 20;
  white-space: nowrap;
}}
.insight-pill {{
  font-size: 0.74rem;
  color: var(--insight-text);
  background: var(--insight-bg);
  border-radius: 8px;
  padding: 5px 10px;
  line-height: 1.35;
}}

/* METRICS STRIP */
.metrics-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}}
.metrics-strip.cols-3 {{
  grid-template-columns: repeat(3, minmax(0, 1fr));
}}
@media (max-width: 640px) {{
  .metrics-strip, .metrics-strip.cols-3 {{ grid-template-columns: repeat(2, 1fr); }}
}}
.m-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 7px 10px;
}}
.m-lbl {{
  font-size: 0.69rem;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.m-val {{
  font-size: 0.98rem;
  font-weight: 700;
  color: var(--text);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.m-delta {{
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--accent);
  margin-left: 5px;
}}

/* COLLAPSIBLE LIVE TABLE */
details.live-details {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 7px 12px;
  font-size: 0.78rem;
}}
details.live-details summary {{
  cursor: pointer;
  font-weight: 600;
  color: var(--text-muted);
}}
table.live-tbl {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 8px;
  font-size: 0.75rem;
}}
table.live-tbl th, table.live-tbl td {{
  text-align: left;
  padding: 5px 7px;
  border-bottom: 1px solid var(--border);
}}
table.live-tbl th {{
  color: var(--text-muted);
  font-weight: 700;
}}
</style>
</head>
<body>
<div class="studio">
  <!-- LEFT: LIVE CONTROLS -->
  <div class="panel controls-panel">
    <div class="controls-title">
      <span>{html.escape(t('parameters_header', lang))}</span>
      <span class="live-badge"><span class="live-dot"></span>LIVE 60 FPS</span>
    </div>
    <div id="slidersContainer"></div>

    {sweep_select_html}
    {dist_view_html}

    <div class="ctrl-block">
      <div class="ctrl-label">{html.escape(t('scaling_mode_label', lang))}</div>
      <div class="seg-group" id="scaleModeSeg">
        <button type="button" class="seg-btn {sens_act}" data-val="sensitivity">⚡ {html.escape(t('scaling_sensitivity', lang).split('/')[0].strip())}</button>
        <button type="button" class="seg-btn {val_act}" data-val="value">📏 {html.escape(t('scaling_value', lang))}</button>
      </div>
    </div>

    <label class="check-row">
      <input type="checkbox" id="colorCheck" {color_checked} />
      <span>{html.escape(t('color_coding_label', lang))}</span>
    </label>

    <div class="ctrl-block">
      <div class="ctrl-label">{html.escape(t('ghost_section', lang))}</div>
      <div class="btn-row">
        <button type="button" class="act-btn" id="btnSaveGhost">📌 {html.escape(t('ghost_save', lang))}</button>
        <button type="button" class="act-btn" id="btnClearGhost">🗑️ {html.escape(t('ghost_clear', lang))}</button>
      </div>
      <div class="ghost-badge" id="ghostBadge">{html.escape(t('ghost_active_note', lang))}</div>
    </div>

    <button type="button" class="act-btn full" id="btnResetBase">↺ {html.escape(t('reset_baseline', lang))}</button>
  </div>

  <!-- RIGHT: FORMULA + CHART + METRICS ALL VISIBLE TOGETHER -->
  <div class="stage-col">
    <div class="panel formula-panel">
      <div class="formula-top-bar">
        <span class="formula-title">{html.escape(t('formula_card_title', lang))}</span>
        <span class="formula-hint" id="formulaHint">{html.escape(t('formula_card_hint_sens' if initial_scaling_mode == 'sensitivity' else 'formula_card_hint_val', lang))}</span>
      </div>
      <div class="fv-math-stage" id="formulaStage">{stage_inner_html}</div>
      <div class="occ-strip" id="occStrip"></div>
    </div>

    <div class="panel chart-panel">
      <div class="chart-header">
        <span class="chart-title" id="chartTitle">{html.escape(title)}</span>
        <span class="chart-hint">{html.escape(t('drag_chart_hint', lang))}</span>
      </div>
      <div class="canvas-wrap">
        <canvas id="liveChart"></canvas>
        <div class="chart-tooltip" id="chartTooltip"></div>
      </div>
      {insight_html}
    </div>

    <div class="metrics-strip {'cols-3' if not is_dist else ''}" id="metricsStrip"></div>

    <details class="live-details">
      <summary>{html.escape(t('live_table_toggle', lang))}</summary>
      <table class="live-tbl">
        <thead>
          <tr>
            <th>{html.escape(t('col_symbol', lang))}</th>
            <th>{html.escape(t('col_context', lang))}</th>
            <th>{html.escape(t('col_value', lang))}</th>
            <th>{html.escape(t('col_partial', lang))}</th>
            <th>{html.escape(t('col_elasticity', lang))}</th>
            <th>{html.escape(t('col_scale', lang))}</th>
          </tr>
        </thead>
        <tbody id="liveTableBody"></tbody>
      </table>
    </details>
  </div>
</div>

<script>
(function() {{
  const CFG = {config_json};

  // --- Self-contained high-precision math runtime ---
  const _M = {{
    logGamma(z) {{
      const g = 7;
      const p = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
      ];
      if (z < 0.5) {{
        return Math.log(Math.PI / Math.sin(Math.PI * z)) - _M.logGamma(1 - z);
      }}
      z -= 1;
      let x = p[0];
      for (let i = 1; i < g + 2; i++) x += p[i] / (z + i);
      const t = z + g + 0.5;
      return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
    }},
    gamma(z) {{
      if (!Number.isFinite(z) || (z <= 0 && Math.abs(z - Math.round(z)) < 1e-12)) return NaN;
      return Math.exp(_M.logGamma(z));
    }},
    binom(n, k) {{
      if (k < 0 || k > n) return 0;
      return Math.exp(_M.logGamma(n + 1) - _M.logGamma(k + 1) - _M.logGamma(n - k + 1));
    }},
    erf(x) {{
      const sign = x < 0 ? -1 : 1;
      const a = Math.abs(x);
      const t = 1.0 / (1.0 + 0.3275911 * a);
      const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
      return sign * y;
    }},
    digamma(x) {{
      if (!Number.isFinite(x) || x <= 0) return NaN;
      let res = 0;
      while (x < 7) {{
        res -= 1 / x;
        x += 1;
      }}
      const inv = 1 / x;
      const inv2 = inv * inv;
      res += Math.log(x) - 0.5 * inv - inv2 * (1/12 - inv2 * (1/120 - inv2 / 252));
      return res;
    }}
  }};

  function compileFn(code) {{
    if (!code) return null;
    try {{
      return new Function("v", "_M", "return (" + code + ");");
    }} catch (e) {{
      return null;
    }}
  }}

  const evalFn = compileFn(CFG.eval_js);
  const taggedEvalFn = compileFn(CFG.tagged_eval_js);
  const elasticityFns = {{}};
  const partialFns = {{}};
  for (const occ of CFG.occurrences) {{
    elasticityFns[occ.occ_id] = compileFn(CFG.elasticity_js[occ.occ_id]);
    partialFns[occ.occ_id] = compileFn(CFG.partial_js[occ.occ_id]);
  }}

  // --- Live state ---
  const state = {{
    values: {{}},
    scalingMode: CFG.scaling_mode,
    colorCoding: CFG.color_coding,
    viewType: CFG.view_type,
    sweepVar: CFG.sweep_var,
    ghost: null,
    baselineOutput: 1.0,
    baselineValues: {{}},
    baselinePartials: {{}},
    baselineYMax: null,
    draggingChart: false,
    hoverPt: null,
    lastPlotMeta: null
  }};

  for (const s of CFG.sliders) {{
    state.values[s.name] = Number(s.value);
    state.baselineValues[s.name] = Number(s.value);
  }}

  function esc(str) {{
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }}

  function fmt(val) {{
    if (val === null || val === undefined || !Number.isFinite(val)) return "—";
    if (Math.abs(val) === 0) return "0";
    if (Math.abs(val) >= 10000 || (Math.abs(val) > 0 && Math.abs(val) < 0.0001)) return val.toExponential(2);
    return Number(val.toPrecision(4)).toString();
  }}

  function fmtTick(val) {{
    if (!Number.isFinite(val)) return "";
    if (Math.abs(val) < 1e-10) return "0";
    if (Math.abs(val) >= 1000 || Math.abs(val) < 0.001) return val.toExponential(1);
    return Number(val.toPrecision(3)).toString();
  }}

  function smoothScale(ratio, minS = 0.62, maxS = 2.02) {{
    if (!Number.isFinite(ratio) || ratio <= 0) return minS;
    const raw = 1.0 + 0.85 * Math.tanh(0.65 * Math.log(Math.max(ratio, 1e-6)));
    return Math.max(minS, Math.min(maxS, raw));
  }}

  function evalTaggedSafe(occVals) {{
    if (!taggedEvalFn) return NaN;
    try {{
      const y = taggedEvalFn(occVals, _M);
      return Number.isFinite(y) ? y : NaN;
    }} catch (e) {{
      return NaN;
    }}
  }}

  function evalPointSafe(vals) {{
    if (CFG.dist_key === "uniform") {{
      const a = vals.a, b = Math.max(a + 0.1, vals.b), x = vals.x;
      if (x < a || x > b) return 0;
      return 1.0 / (b - a);
    }}
    if (!evalFn) return NaN;
    try {{
      const y = evalFn(vals, _M);
      return Number.isFinite(y) ? y : NaN;
    }} catch (e) {{
      return NaN;
    }}
  }}

  function numericalPartial(occVals, oid) {{
    const x0 = occVals[oid];
    const h = Math.max(1e-5, Math.abs(x0) * 1e-5);
    const vp = Object.assign({{}}, occVals); vp[oid] = x0 + h;
    const vm = Object.assign({{}}, occVals); vm[oid] = x0 - h;
    const fp = evalTaggedSafe(vp);
    const fm = evalTaggedSafe(vm);
    if (Number.isFinite(fp) && Number.isFinite(fm)) return (fp - fm) / (2 * h);
    const f0 = evalTaggedSafe(occVals);
    if (Number.isFinite(fp) && Number.isFinite(f0)) return (fp - f0) / h;
    if (Number.isFinite(fm) && Number.isFinite(f0)) return (f0 - fm) / h;
    return 0;
  }}

  function evaluateCurrent() {{
    const vals = Object.assign({{}}, state.values);
    if (CFG.dist_key === "uniform" && vals.b <= vals.a + 0.1) {{
      vals.b = vals.a + 0.2;
    }}
    const occVals = {{}};
    for (const occ of CFG.occurrences) {{
      occVals[occ.occ_id] = vals[occ.var_name];
    }}
    const outVal = evalTaggedSafe(occVals);
    const valid = Number.isFinite(outVal);

    const refOut = (Number.isFinite(state.baselineOutput) && Math.abs(state.baselineOutput) > 1e-12) ? state.baselineOutput : 1.0;
    const responseScale = !valid ? 1.0 : (Math.abs(outVal) < 1e-12 ? 0.68 : smoothScale(Math.abs(outVal) / Math.abs(refOut), 0.65, 2.02));

    const nonInd = CFG.occurrences.filter(o => !o.is_independent);
    const singleParam = nonInd.length === 1;
    const rawPartials = {{}};
    const rawElast = {{}};

    for (const occ of CFG.occurrences) {{
      const oid = occ.occ_id;
      const vi = occVals[oid];
      let pVal = NaN;
      if (partialFns[oid]) {{
        try {{ pVal = partialFns[oid](occVals, _M); }} catch (e) {{ pVal = NaN; }}
      }}
      if (!Number.isFinite(pVal)) pVal = numericalPartial(occVals, oid);
      if (!Number.isFinite(pVal)) pVal = 0;
      rawPartials[oid] = pVal;

      let eVal = NaN;
      if (elasticityFns[oid]) {{
        try {{ eVal = elasticityFns[oid](occVals, _M); }} catch (e) {{ eVal = NaN; }}
      }}
      if (Math.abs(vi) < 1e-6) {{
        eVal = (valid && Math.abs(outVal) > 1e-15) ? (pVal / Math.abs(outVal)) : pVal;
      }} else if (!Number.isFinite(eVal)) {{
        eVal = (valid && Math.abs(outVal) > 1e-15) ? ((vi * pVal) / outVal) : (vi * pVal);
      }}
      if (!Number.isFinite(eVal)) eVal = 0;
      rawElast[oid] = eVal;
    }}

    let totalEff = 0;
    for (const occ of CFG.occurrences) totalEff += Math.abs(rawElast[occ.occ_id]);
    if (totalEff < 1e-12) totalEff = 1.0;

    const results = {{}};
    for (const occ of CFG.occurrences) {{
      const oid = occ.occ_id;
      const vi = occVals[oid];
      const pVal = rawPartials[oid];
      const eVal = rawElast[oid];
      const dir = pVal > 1e-9 ? "pos" : (pVal < -1e-9 ? "neg" : "neutral");

      let scale = 1.0;
      if (state.scalingMode === "value") {{
        const refV = Math.abs(state.baselineValues[occ.var_name] || 1.0) > 1e-9 ? Math.abs(state.baselineValues[occ.var_name]) : 1.0;
        scale = Math.abs(vi) < 1e-9 ? 0.65 : smoothScale(Math.abs(vi) / refV);
      }} else {{
        if (singleParam && !occ.is_independent) {{
          const refP = Math.abs(state.baselinePartials[oid] || 1.0) > 1e-12 ? Math.abs(state.baselinePartials[oid]) : 1.0;
          scale = smoothScale(Math.abs(pVal) / refP);
        }} else {{
          scale = Math.abs(eVal) < 1e-6 ? 0.65 : smoothScale(Math.abs(eVal));
        }}
      }}

      results[oid] = {{
        occ, value: vi, partial: pVal, elasticity: eVal, direction: dir, scale,
        share: Math.abs(eVal) / totalEff
      }};
    }}

    return {{ outVal, valid, responseScale, results, rawPartials }};
  }}

  function captureBaseline() {{
    const probe = evaluateCurrent();
    state.baselineOutput = (probe.valid && Math.abs(probe.outVal) > 1e-12) ? probe.outVal : 1.0;
    state.baselineValues = Object.assign({{}}, state.values);
    state.baselinePartials = Object.assign({{}}, probe.rawPartials);
    state.baselineYMax = null;
  }}

  function computeDistStats(vals) {{
    const k = CFG.dist_key;
    if (!k) return null;
    let mean = NaN, variance = NaN;
    if (k === "normal") {{ mean = vals.mu; variance = vals.sigma * vals.sigma; }}
    else if (k === "gamma") {{ mean = vals.alpha / vals.beta; variance = vals.alpha / (vals.beta * vals.beta); }}
    else if (k === "binomial") {{ const n = Math.round(vals.n); mean = n * vals.p; variance = n * vals.p * (1 - vals.p); }}
    else if (k === "poisson") {{ mean = vals.lam; variance = vals.lam; }}
    else if (k === "exponential") {{ mean = 1 / vals.lam; variance = 1 / (vals.lam * vals.lam); }}
    else if (k === "beta") {{ const s = vals.alpha + vals.beta; mean = vals.alpha / s; variance = (vals.alpha * vals.beta) / (s * s * (s + 1)); }}
    else if (k === "student_t") {{ mean = vals.nu > 1 ? 0 : NaN; variance = vals.nu > 2 ? vals.nu / (vals.nu - 2) : Infinity; }}
    else if (k === "chi2") {{ mean = vals.k; variance = 2 * vals.k; }}
    else if (k === "lognormal") {{ const s2 = vals.sigma * vals.sigma; mean = Math.exp(vals.mu + s2 / 2); variance = (Math.exp(s2) - 1) * Math.exp(2 * vals.mu + s2); }}
    else if (k === "uniform") {{ const a = vals.a, b = Math.max(a + 0.1, vals.b); mean = 0.5 * (a + b); variance = Math.pow(b - a, 2) / 12; }}
    const std = (Number.isFinite(variance) && variance >= 0) ? Math.sqrt(variance) : NaN;
    return {{ mean, variance, std }};
  }}

  function getCurveData() {{
    const vals = Object.assign({{}}, state.values);
    if (CFG.dist_key === "uniform" && vals.b <= vals.a + 0.1) vals.b = vals.a + 0.2;
    const isDist = CFG.mode_kind === "continuous_dist" || CFG.mode_kind === "discrete_dist";
    const isDisc = CFG.mode_kind === "discrete_dist" || (CFG.mode_kind === "discrete_func" && state.sweepVar === CFG.independent_var);
    const xVar = isDist ? CFG.independent_var : state.sweepVar;
    if (!xVar) return null;

    let xMin = -5, xMax = 5;
    if (isDist) {{
      const k = CFG.dist_key;
      if (k === "normal") {{ xMin = Math.min(-6, vals.mu - 4 * vals.sigma); xMax = Math.max(6, vals.mu + 4 * vals.sigma); }}
      else if (k === "gamma") {{ xMin = 0.01; xMax = Math.max(10, (vals.alpha / vals.beta) + 4.5 * Math.sqrt(vals.alpha) / vals.beta); }}
      else if (k === "binomial") {{ xMin = 0; xMax = Math.round(vals.n); }}
      else if (k === "poisson") {{ xMin = 0; xMax = Math.max(15, Math.ceil(vals.lam + 4.5 * Math.sqrt(vals.lam))); }}
      else if (k === "exponential") {{ xMin = 0; xMax = Math.max(6, 5.2 / vals.lam); }}
      else if (k === "beta") {{ xMin = 0.005; xMax = 0.995; }}
      else if (k === "student_t") {{ xMin = -6; xMax = 6; }}
      else if (k === "chi2") {{ xMin = 0.05; xMax = Math.max(12, vals.k + 4.2 * Math.sqrt(2 * vals.k)); }}
      else if (k === "lognormal") {{ xMin = 0.02; xMax = Math.min(35, Math.max(6, Math.exp(vals.mu + 2.4 * vals.sigma))); }}
      else if (k === "uniform") {{ xMin = vals.a - 1.5; xMax = vals.b + 1.5; }}
    }} else {{
      const sl = CFG.sliders.find(s => s.name === xVar);
      xMin = sl ? sl.min : -5;
      xMax = sl ? sl.max : 5;
    }}

    const xs = [], ys = [];
    if (isDisc) {{
      for (let k = Math.floor(xMin); k <= Math.ceil(xMax); k++) {{
        xs.push(k);
        const pv = Object.assign({{}}, vals); pv[xVar] = k;
        ys.push(evalPointSafe(pv));
      }}
      if (isDist && state.viewType === "cdf") {{
        let acc = 0;
        for (let i = 0; i < ys.length; i++) {{
          acc += Number.isFinite(ys[i]) ? ys[i] : 0;
          ys[i] = Math.min(1, acc);
        }}
      }}
    }} else {{
      const N = 260;
      for (let i = 0; i < N; i++) {{
        const x = xMin + (xMax - xMin) * (i / (N - 1));
        xs.push(x);
        const pv = Object.assign({{}}, vals); pv[xVar] = x;
        ys.push(evalPointSafe(pv));
      }}
      if (isDist && state.viewType === "cdf") {{
        const k = CFG.dist_key;
        if (k === "normal") {{
          for (let i = 0; i < N; i++) ys[i] = 0.5 * (1 + _M.erf((xs[i] - vals.mu) / (vals.sigma * Math.SQRT2)));
        }} else if (k === "lognormal") {{
          for (let i = 0; i < N; i++) ys[i] = xs[i] <= 0 ? 0 : 0.5 * (1 + _M.erf((Math.log(xs[i]) - vals.mu) / (vals.sigma * Math.SQRT2)));
        }} else if (k === "exponential") {{
          for (let i = 0; i < N; i++) ys[i] = xs[i] < 0 ? 0 : 1 - Math.exp(-vals.lam * xs[i]);
        }} else if (k === "uniform") {{
          for (let i = 0; i < N; i++) ys[i] = xs[i] <= vals.a ? 0 : (xs[i] >= vals.b ? 1 : (xs[i] - vals.a) / (vals.b - vals.a));
        }} else {{
          // Accurate trapezoidal cumulative integration of PDF
          const cdf = new Array(N).fill(0);
          let area = 0;
          for (let i = 1; i < N; i++) {{
            const y0 = Number.isFinite(ys[i - 1]) ? Math.max(0, ys[i - 1]) : 0;
            const y1 = Number.isFinite(ys[i]) ? Math.max(0, ys[i]) : 0;
            area += 0.5 * (y0 + y1) * (xs[i] - xs[i - 1]);
            cdf[i] = Math.min(1.0, area);
          }}
          if (area > 0.05 && k === "student_t") {{
            // Shift symmetric Student-t so F(0) = 0.5
            const mid = cdf[Math.floor(N / 2)];
            const shift = 0.5 - mid;
            for (let i = 0; i < N; i++) cdf[i] = Math.max(0, Math.min(1, cdf[i] + shift));
          }}
          for (let i = 0; i < N; i++) ys[i] = cdf[i];
        }}
      }}
    }}

    // Compute focus y matching plot mode (CDF vs PDF/PMF/func)
    const focusX = vals[xVar];
    let focusY = evalPointSafe(vals);
    if (isDist && state.viewType === "cdf") {{
      // Interpolate from xs/ys
      if (isDisc) {{
        const idx = xs.findIndex(v => Math.abs(v - Math.round(focusX)) < 0.5);
        focusY = idx >= 0 ? ys[idx] : NaN;
      }} else {{
        const tIdx = ((focusX - xMin) / (xMax - xMin)) * (xs.length - 1);
        const i0 = Math.max(0, Math.min(xs.length - 2, Math.floor(tIdx)));
        const frac = Math.max(0, Math.min(1, tIdx - i0));
        focusY = ys[i0] + frac * (ys[i0 + 1] - ys[i0]);
      }}
    }}

    return {{ xs, ys, xMin, xMax, isDisc, isDist, xVar, focusX, focusY }};
  }}

  // --- Update DOM Formula, Chips, Metrics, Table ---
  function updateFormulaDOM(evalRes) {{
    const stage = document.getElementById("formulaStage");
    const baseRem = 1.32;

    const lhsEl = stage.querySelector('[data-role="lhs"]');
    if (lhsEl) {{
      lhsEl.style.fontSize = (baseRem * 1.03 * evalRes.responseScale).toFixed(3) + "rem";
      lhsEl.title = `${{CFG.lhs_display}}\\n${{CFG.strings.col_value}}: ${{fmt(evalRes.outVal)}}\\n${{CFG.strings.col_scale}}: ${{evalRes.responseScale.toFixed(2)}}×`;
    }}
    const rhsEl = stage.querySelector('[data-role="rhs-value"]');
    if (rhsEl) {{
      rhsEl.textContent = "= " + (evalRes.valid ? fmt(evalRes.outVal) : "—");
    }}

    const hintEl = document.getElementById("formulaHint");
    if (hintEl) {{
      hintEl.textContent = state.scalingMode === "sensitivity" ? CFG.strings.hint_sens : CFG.strings.hint_val;
    }}

    const chips = [];
    const tblRows = [];

    for (const occ of CFG.occurrences) {{
      const r = evalRes.results[occ.occ_id];
      const tokEl = stage.querySelector(`[data-occ-id="${{occ.occ_id}}"]`);
      if (tokEl && r) {{
        tokEl.style.fontSize = (baseRem * r.scale).toFixed(3) + "rem";
        tokEl.classList.remove("fv-tok-pos", "fv-tok-neg", "fv-tok-neutral", "fv-tok-mono");
        if (state.colorCoding) {{
          tokEl.classList.add(r.direction === "pos" ? "fv-tok-pos" : (r.direction === "neg" ? "fv-tok-neg" : "fv-tok-neutral"));
        }} else {{
          tokEl.classList.add("fv-tok-mono");
        }}
        const dirText = r.direction === "pos" ? CFG.strings.dir_up : (r.direction === "neg" ? CFG.strings.dir_down : CFG.strings.dir_neutral);
        const occTitle = occ.total_occurrences > 1
          ? `${{occ.display_name}} (${{CFG.strings.occ_tpl.replace("{{idx}}", occ.occurrence_index)}} — ${{occ.context}})`
          : `${{occ.display_name}} (${{occ.context}})`;
        tokEl.title = `${{occTitle}}\\n${{CFG.strings.col_value}}: ${{fmt(r.value)}}\\n${{CFG.strings.col_partial}}: ${{fmt(r.partial)}}\\n${{CFG.strings.col_elasticity}}: ${{fmt(r.elasticity)}}\\n${{CFG.strings.col_direction}}: ${{dirText}}\\n${{CFG.strings.col_scale}}: ${{r.scale.toFixed(2)}}×`;
      }}

      if (r) {{
        const dotColor = state.colorCoding
          ? (r.direction === "pos" ? "var(--pos)" : (r.direction === "neg" ? "var(--neg)" : "var(--neu)"))
          : "var(--accent)";
        const arrow = state.colorCoding
          ? (r.direction === "pos" ? "↑" : (r.direction === "neg" ? "↓" : "→"))
          : "•";
        const idxTag = occ.total_occurrences > 1 ? ` #${{occ.occurrence_index}}` : "";
        chips.push(
          `<span class="occ-chip" title="${{esc(occ.context)}} (E = ${{fmt(r.elasticity)}})">` +
          `<span class="occ-chip-dot" style="background:${{dotColor}}"></span>` +
          `<strong>${{esc(occ.display_name)}}${{idxTag}}</strong> ` +
          `<span style="color:${{dotColor}};font-weight:700">${{arrow}} ${{r.scale.toFixed(2)}}×</span>` +
          `<span style="color:var(--text-muted)">(${{Math.round(r.share * 100)}}%)</span>` +
          `</span>`
        );
        tblRows.push(
          `<tr><td><strong>${{esc(occ.display_name)}}${{idxTag}}</strong></td>` +
          `<td>${{esc(occ.context)}}</td><td>${{fmt(r.value)}}</td>` +
          `<td>${{fmt(r.partial)}}</td><td>${{fmt(r.elasticity)}}</td>` +
          `<td>${{r.scale.toFixed(2)}}×</td></tr>`
        );
      }}
    }}

    document.getElementById("occStrip").innerHTML = chips.join("");
    document.getElementById("liveTableBody").innerHTML = tblRows.join("");
  }}

  function updateMetricsDOM(evalRes, curve) {{
    const strip = document.getElementById("metricsStrip");
    const isDist = CFG.mode_kind === "continuous_dist" || CFG.mode_kind === "discrete_dist";
    if (isDist) {{
      const st = computeDistStats(state.values);
      const gst = state.ghost ? state.ghost.stats : null;
      const deltaStr = (a, b) => {{
        if (!gst || !Number.isFinite(a) || !Number.isFinite(b)) return "";
        const d = a - b;
        if (Math.abs(d) < 1e-7) return "";
        return `<span class="m-delta">${{d > 0 ? "+" : ""}}${{fmt(d)}}</span>`;
      }};
      const respLbl = `${{CFG.strings.metric_response}} (${{CFG.independent_var}}₀=${{fmt(curve.focusX)}})`;
      strip.innerHTML =
        `<div class="m-card"><div class="m-lbl">${{respLbl}}</div><div class="m-val">${{fmt(curve.focusY)}}</div></div>` +
        `<div class="m-card"><div class="m-lbl">${{CFG.strings.metric_mean}}</div><div class="m-val">${{fmt(st.mean)}}${{deltaStr(st.mean, gst && gst.mean)}}</div></div>` +
        `<div class="m-card"><div class="m-lbl">${{CFG.strings.metric_var}}</div><div class="m-val">${{fmt(st.variance)}}${{deltaStr(st.variance, gst && gst.variance)}}</div></div>` +
        `<div class="m-card"><div class="m-lbl">${{CFG.strings.metric_std}}</div><div class="m-val">${{fmt(st.std)}}${{deltaStr(st.std, gst && gst.std)}}</div></div>`;
    }} else {{
      let dom = "—";
      let maxE = -1;
      for (const occ of CFG.occurrences) {{
        const r = evalRes.results[occ.occ_id];
        if (r && Math.abs(r.elasticity) > maxE) {{
          maxE = Math.abs(r.elasticity);
          dom = occ.total_occurrences > 1 ? `${{occ.display_name}} #${{occ.occurrence_index}} (${{occ.context}})` : occ.display_name;
        }}
      }}
      strip.innerHTML =
        `<div class="m-card"><div class="m-lbl">${{CFG.strings.metric_scalar}} (${{CFG.lhs_display}})</div><div class="m-val">${{evalRes.valid ? fmt(evalRes.outVal) : "—"}}</div></div>` +
        `<div class="m-card"><div class="m-lbl">${{CFG.strings.metric_type}}</div><div class="m-val" style="font-family:inherit;font-size:0.84rem">${{CFG.output_type_label}}</div></div>` +
        `<div class="m-card"><div class="m-lbl">${{CFG.strings.metric_dom}}</div><div class="m-val" style="font-family:inherit;font-size:0.84rem">${{dom}}</div></div>`;
    }}
  }}

  // --- Retina HTML5 Canvas 60 FPS Chart Renderer ---
  const canvas = document.getElementById("liveChart");
  const ctx = canvas.getContext("2d");

  function drawChart(curve) {{
    if (!curve) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rect.width;
    const H = rect.height;
    const pad = {{ l: 50, r: 20, t: 22, b: 36 }};
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const cs = getComputedStyle(document.documentElement);
    const colAccent = cs.getPropertyValue("--accent").trim();
    const colText = cs.getPropertyValue("--text").trim();
    const colMuted = cs.getPropertyValue("--text-muted").trim();
    const colGrid = cs.getPropertyValue("--grid").trim();
    const colFocus = cs.getPropertyValue("--focus").trim();
    const colMean = cs.getPropertyValue("--mean").trim();
    const isDark = document.documentElement.classList.contains("dark");

    ctx.clearRect(0, 0, W, H);

    // Compute robust Y bounds (handling asymptotes like 1/a cleanly!)
    const finiteYs = curve.ys.filter(y => Number.isFinite(y));
    let yMin = 0, yMax = 1;
    if (curve.isDist && state.viewType === "cdf") {{
      yMin = 0; yMax = 1.05;
    }} else if (finiteYs.length > 0) {{
      const sorted = [...finiteYs].sort((a, b) => a - b);
      const q02 = sorted[Math.floor(sorted.length * 0.02)];
      const q98 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))];
      let rawMin = curve.isDist ? 0 : Math.min(0, q02);
      let rawMax = Math.max(q98, Number.isFinite(curve.focusY) ? curve.focusY : q98);
      if (state.ghost && state.ghost.ys) {{
        for (const gy of state.ghost.ys) {{
          if (Number.isFinite(gy) && gy < rawMax * 5 + 10) rawMax = Math.max(rawMax, gy);
        }}
      }}
      // Keep stable baseline ceiling for distributions so dragging sigma visibly grows/shrinks peak!
      if (curve.isDist) {{
        if (state.baselineYMax === null) state.baselineYMax = rawMax * 1.18;
        rawMax = Math.max(rawMax * 1.12, state.baselineYMax);
      }} else {{
        const span = Math.max(1e-6, rawMax - rawMin);
        rawMax += span * 0.12;
        if (rawMin < 0) rawMin -= span * 0.08;
      }}
      yMin = rawMin;
      yMax = Math.max(rawMin + 1e-6, rawMax);
    }}

    const xMin = curve.isDisc ? curve.xMin - 0.6 : curve.xMin;
    const xMax = curve.isDisc ? curve.xMax + 0.6 : curve.xMax;
    const mapX = x => pad.l + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * pw;
    const mapY = y => pad.t + ph - ((y - yMin) / Math.max(1e-9, yMax - yMin)) * ph;
    const unmapX = px => xMin + ((px - pad.l) / pw) * (xMax - xMin);

    state.lastPlotMeta = {{ pad, pw, ph, xMin, xMax, yMin, yMax, mapX, mapY, unmapX, curve }};

    // Grid & axes
    ctx.strokeStyle = colGrid;
    ctx.lineWidth = 1;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = colMuted;

    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {{
      const yv = yMin + (yMax - yMin) * (i / yTicks);
      const py = mapY(yv);
      ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + pw, py); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(fmtTick(yv), pad.l - 7, py);
    }}
    const xTicks = 6;
    for (let i = 0; i <= xTicks; i++) {{
      const xv = curve.xMin + (curve.xMax - curve.xMin) * (i / xTicks);
      const px = mapX(xv);
      ctx.beginPath(); ctx.moveTo(px, pad.t); ctx.lineTo(px, pad.t + ph); ctx.stroke();
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(fmtTick(xv), px, pad.t + ph + 7);
    }}

    // Axis title
    const xLabel = CFG.display_names[curve.xVar] || curve.xVar;
    ctx.fillStyle = colMuted;
    ctx.font = "600 11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(xLabel, pad.l + pw / 2, H - 12);

    // Clip plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, pw, ph);
    ctx.clip();

    // 1. Ghost reference curve
    if (state.ghost && state.ghost.viewType === state.viewType && state.ghost.xVar === curve.xVar) {{
      const gx = state.ghost.xs, gy = state.ghost.ys;
      if (curve.isDisc && state.viewType !== "cdf") {{
        const bw = Math.max(4, Math.min(24, (pw / Math.max(1, gx.length)) * 0.56));
        ctx.fillStyle = isDark ? "rgba(148,163,184,0.22)" : "rgba(148,163,184,0.32)";
        ctx.strokeStyle = "#94A3B8";
        ctx.lineWidth = 1.2;
        for (let i = 0; i < gx.length; i++) {{
          if (!Number.isFinite(gy[i])) continue;
          const bx = mapX(gx[i]) - bw / 2;
          const by = mapY(gy[i]);
          const b0 = mapY(0);
          ctx.fillRect(bx, by, bw, b0 - by);
          ctx.strokeRect(bx, by, bw, b0 - by);
        }}
      }} else {{
        ctx.beginPath();
        ctx.setLineDash([6, 5]);
        ctx.strokeStyle = "#94A3B8";
        ctx.lineWidth = 2.0;
        let started = false;
        for (let i = 0; i < gx.length; i++) {{
          if (!Number.isFinite(gy[i])) {{ started = false; continue; }}
          const px = mapX(gx[i]), py = mapY(gy[i]);
          if (!started) {{ ctx.moveTo(px, py); started = true; }}
          else {{
            if (curve.isDisc && state.viewType === "cdf") ctx.lineTo(px, mapY(gy[i - 1]));
            ctx.lineTo(px, py);
          }}
        }}
        ctx.stroke();
        ctx.setLineDash([]);
      }}
    }}

    // 2. Main current curve / bars
    if (curve.isDisc && state.viewType !== "cdf") {{
      const bw = Math.max(5, Math.min(26, (pw / Math.max(1, curve.xs.length)) * 0.62));
      const b0 = mapY(0);
      for (let i = 0; i < curve.xs.length; i++) {{
        if (!Number.isFinite(curve.ys[i])) continue;
        const isFocusBar = Math.abs(curve.xs[i] - Math.round(curve.focusX)) < 0.5;
        ctx.fillStyle = isFocusBar ? colFocus : colAccent;
        ctx.globalAlpha = isFocusBar ? 0.95 : 0.82;
        const bx = mapX(curve.xs[i]) - bw / 2;
        const by = mapY(curve.ys[i]);
        ctx.fillRect(bx, by, bw, b0 - by);
      }}
      ctx.globalAlpha = 1.0;
    }} else {{
      // Filled area for continuous PDF
      if (curve.isDist && state.viewType !== "cdf") {{
        ctx.beginPath();
        let started = false, firstX = 0, lastX = 0;
        for (let i = 0; i < curve.xs.length; i++) {{
          if (!Number.isFinite(curve.ys[i])) continue;
          const px = mapX(curve.xs[i]), py = mapY(curve.ys[i]);
          if (!started) {{ ctx.moveTo(px, mapY(0)); ctx.lineTo(px, py); firstX = px; started = true; }}
          else {{ ctx.lineTo(px, py); }}
          lastX = px;
        }}
        if (started) {{
          ctx.lineTo(lastX, mapY(0));
          ctx.closePath();
          const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ph);
          grad.addColorStop(0, isDark ? "rgba(96,165,250,0.34)" : "rgba(37,99,235,0.22)");
          grad.addColorStop(1, isDark ? "rgba(96,165,250,0.02)" : "rgba(37,99,235,0.01)");
          ctx.fillStyle = grad;
          ctx.fill();
        }}
      }}
      // Stroke line
      ctx.beginPath();
      ctx.strokeStyle = colAccent;
      ctx.lineWidth = 2.8;
      let started = false;
      for (let i = 0; i < curve.xs.length; i++) {{
        if (!Number.isFinite(curve.ys[i])) {{ started = false; continue; }}
        const px = mapX(curve.xs[i]), py = mapY(curve.ys[i]);
        if (!started) {{ ctx.moveTo(px, py); started = true; }}
        else {{
          if (curve.isDisc && state.viewType === "cdf") ctx.lineTo(px, mapY(curve.ys[i - 1]));
          ctx.lineTo(px, py);
        }}
      }}
      ctx.stroke();
    }}

    // 3. Mean line for distributions
    if (curve.isDist) {{
      const st = computeDistStats(state.values);
      if (st && Number.isFinite(st.mean) && st.mean >= xMin && st.mean <= xMax) {{
        const mx = mapX(st.mean);
        ctx.beginPath();
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = colMean;
        ctx.lineWidth = 1.6;
        ctx.moveTo(mx, pad.t);
        ctx.lineTo(mx, pad.t + ph);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = colMean;
        ctx.font = "600 11px -apple-system, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`μ=${{fmtTick(st.mean)}}`, Math.min(mx + 4, pad.l + pw - 56), pad.t + 10);
      }}
    }}

    // 4. Focus point vertical guide & glowing dot
    if (Number.isFinite(curve.focusX) && Number.isFinite(curve.focusY)) {{
      const fx = mapX(curve.focusX);
      const fy = mapY(curve.focusY);
      ctx.beginPath();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = colFocus;
      ctx.lineWidth = 1.8;
      ctx.moveTo(fx, pad.t);
      ctx.lineTo(fx, pad.t + ph);
      ctx.stroke();
      ctx.setLineDash([]);

      // Glow ring + solid dot
      ctx.beginPath();
      ctx.arc(fx, fy, 9, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(245, 158, 11, 0.25)";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(fx, fy, 5.5, 0, 2 * Math.PI);
      ctx.fillStyle = colFocus;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = isDark ? "#0B1120" : "#FFFFFF";
      ctx.stroke();

      // Callout label
      const tag = `(${{fmt(curve.focusX)}}, ${{fmt(curve.focusY)}})`;
      ctx.font = "700 11px ui-monospace, SFMono-Regular, monospace";
      const tw = ctx.measureText(tag).width + 10;
      const bx = Math.max(pad.l + 4, Math.min(pad.l + pw - tw - 4, fx - tw / 2));
      const by = Math.max(pad.t + 4, fy - 26);
      ctx.fillStyle = isDark ? "rgba(17,24,39,0.92)" : "rgba(255,255,255,0.92)";
      ctx.strokeStyle = colFocus;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, by, tw, 18, 5);
      else ctx.rect(bx, by, tw, 18);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = colFocus;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(tag, bx + tw / 2, by + 9.5);
    }}

    ctx.restore();

    // Update chart title
    const titleEl = document.getElementById("chartTitle");
    if (titleEl) {{
      if (curve.isDist) {{
        const modeTag = state.viewType === "cdf" ? "CDF" : (curve.isDisc ? "PMF" : "PDF");
        titleEl.textContent = `${{CFG.title}} — ${{modeTag}}`;
      }} else {{
        titleEl.textContent = `${{CFG.lhs_display}} vs. ${{xLabel}}`;
      }}
    }}
  }}

  // --- Full 60 FPS Render Pipeline ---
  let rafPending = false;
  function scheduleRender() {{
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {{
      rafPending = false;
      const evalRes = evaluateCurrent();
      const curve = getCurveData();
      updateFormulaDOM(evalRes);
      updateMetricsDOM(evalRes, curve);
      drawChart(curve);
    }});
  }}

  // --- Build Sliders ---
  const sliderRefs = {{}};
  function buildSliders() {{
    const box = document.getElementById("slidersContainer");
    box.innerHTML = "";
    for (const s of CFG.sliders) {{
      const row = document.createElement("div");
      row.className = "slider-row" + (s.is_focus ? " focus-row" : "");

      const head = document.createElement("div");
      head.className = "slider-head";

      const lbl = document.createElement("span");
      lbl.className = "slider-label";
      lbl.textContent = s.label;
      lbl.title = s.label;

      const num = document.createElement("input");
      num.type = "number";
      num.className = "slider-num";
      num.min = s.min;
      num.max = s.max;
      num.step = s.step;
      num.value = state.values[s.name];

      const rng = document.createElement("input");
      rng.type = "range";
      rng.min = s.min;
      rng.max = s.max;
      rng.step = s.step;
      rng.value = state.values[s.name];

      const onValChange = (raw) => {{
        let v = parseFloat(raw);
        if (!Number.isFinite(v)) return;
        if (s.is_int) v = Math.round(v);
        state.values[s.name] = v;
        // Dynamically adjust Binomial focus max when n changes
        if (CFG.dist_key === "binomial" && s.name === "n" && sliderRefs.k) {{
          sliderRefs.k.rng.max = v;
          sliderRefs.k.num.max = v;
          if (state.values.k > v) {{
            state.values.k = v;
            sliderRefs.k.rng.value = v;
            sliderRefs.k.num.value = v;
          }}
        }}
        scheduleRender();
      }};

      rng.addEventListener("input", () => {{
        num.value = rng.value;
        onValChange(rng.value);
      }});
      num.addEventListener("input", () => {{
        rng.value = num.value;
        onValChange(num.value);
      }});

      head.appendChild(lbl);
      head.appendChild(num);
      row.appendChild(head);
      row.appendChild(rng);
      box.appendChild(row);
      sliderRefs[s.name] = {{ rng, num, spec: s }};
    }}
  }}

  // --- Direct Click & Drag on Chart to scrub focus x0 at 60 FPS! ---
  function scrubFromPointer(clientX) {{
    const meta = state.lastPlotMeta;
    if (!meta || !meta.curve) return;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const xVar = meta.curve.xVar;
    const ref = sliderRefs[xVar];
    if (!ref) return;
    let xVal = meta.unmapX(px);
    xVal = Math.max(parseFloat(ref.rng.min), Math.min(parseFloat(ref.rng.max), xVal));
    if (ref.spec.is_int) xVal = Math.round(xVal);
    else {{
      const step = parseFloat(ref.spec.step) || 0.05;
      xVal = Math.round(xVal / step) * step;
      xVal = Number(xVal.toFixed(4));
    }}
    state.values[xVar] = xVal;
    ref.rng.value = xVal;
    ref.num.value = xVal;
    scheduleRender();
  }}

  canvas.addEventListener("pointerdown", e => {{
    state.draggingChart = true;
    canvas.setPointerCapture(e.pointerId);
    scrubFromPointer(e.clientX);
  }});
  canvas.addEventListener("pointermove", e => {{
    if (state.draggingChart) {{
      scrubFromPointer(e.clientX);
    }}
    const meta = state.lastPlotMeta;
    const tip = document.getElementById("chartTooltip");
    if (!meta || !meta.curve) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    if (px >= meta.pad.l && px <= meta.pad.l + meta.pw && py >= meta.pad.t && py <= meta.pad.t + meta.ph) {{
      const xv = meta.unmapX(px);
      const pv = Object.assign({{}}, state.values);
      pv[meta.curve.xVar] = meta.curve.isDisc ? Math.round(xv) : xv;
      let yv = evalPointSafe(pv);
      if (meta.curve.isDist && state.viewType === "cdf") {{
        const c = meta.curve;
        if (c.isDisc) {{
          const idx = c.xs.findIndex(v => Math.abs(v - pv[c.xVar]) < 0.5);
          yv = idx >= 0 ? c.ys[idx] : NaN;
        }} else {{
          const tIdx = ((pv[c.xVar] - c.xMin) / Math.max(1e-9, c.xMax - c.xMin)) * (c.xs.length - 1);
          const i0 = Math.max(0, Math.min(c.xs.length - 2, Math.floor(tIdx)));
          const frac = Math.max(0, Math.min(1, tIdx - i0));
          yv = c.ys[i0] + frac * (c.ys[i0 + 1] - c.ys[i0]);
        }}
      }}
      tip.style.display = "block";
      tip.style.left = Math.min(rect.width - 140, px + 12) + "px";
      tip.style.top = Math.max(4, py - 28) + "px";
      const xName = CFG.display_names[meta.curve.xVar] || meta.curve.xVar;
      tip.textContent = `${{xName}}=${{fmt(pv[meta.curve.xVar])}} | y=${{fmt(yv)}}`;
    }} else {{
      tip.style.display = "none";
    }}
  }});
  canvas.addEventListener("pointerup", () => {{ state.draggingChart = false; }});
  canvas.addEventListener("pointerleave", () => {{
    state.draggingChart = false;
    document.getElementById("chartTooltip").style.display = "none";
  }});

  // --- Wire Controls ---
  document.querySelectorAll("#scaleModeSeg .seg-btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      document.querySelectorAll("#scaleModeSeg .seg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.scalingMode = btn.dataset.val;
      scheduleRender();
    }});
  }});

  document.querySelectorAll("#viewTypeSeg .seg-btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      document.querySelectorAll("#viewTypeSeg .seg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.viewType = btn.dataset.val;
      state.baselineYMax = null;
      scheduleRender();
    }});
  }});

  document.getElementById("colorCheck").addEventListener("change", e => {{
    state.colorCoding = e.target.checked;
    scheduleRender();
  }});

  const sweepSel = document.getElementById("sweepSelect");
  if (sweepSel) {{
    sweepSel.addEventListener("change", () => {{
      state.sweepVar = sweepSel.value;
      state.ghost = null;
      document.getElementById("ghostBadge").classList.remove("show");
      scheduleRender();
    }});
  }}

  document.getElementById("btnSaveGhost").addEventListener("click", () => {{
    const curve = getCurveData();
    if (!curve) return;
    state.ghost = {{
      xs: [...curve.xs],
      ys: [...curve.ys],
      viewType: state.viewType,
      xVar: curve.xVar,
      stats: computeDistStats(state.values)
    }};
    document.getElementById("ghostBadge").classList.add("show");
    scheduleRender();
  }});

  document.getElementById("btnClearGhost").addEventListener("click", () => {{
    state.ghost = null;
    document.getElementById("ghostBadge").classList.remove("show");
    scheduleRender();
  }});

  document.getElementById("btnResetBase").addEventListener("click", () => {{
    captureBaseline();
    scheduleRender();
  }});

  if (typeof ResizeObserver !== "undefined") {{
    new ResizeObserver(() => scheduleRender()).observe(canvas.parentElement);
  }}
  window.addEventListener("resize", () => scheduleRender());

  // Initialize!
  buildSliders();
  captureBaseline();
  scheduleRender();
}})();
</script>
</body>
</html>
"""
