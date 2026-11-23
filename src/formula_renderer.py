"""Dynamic HTML/CSS mathematical formula renderer with per-occurrence visual scaling and tooltips."""

from __future__ import annotations

import html
import math
from typing import Dict, List, Optional, Tuple

import sympy as sp

from src.formula_engine import EvaluationResult, OccurrenceAnalysis, ParsedFormula
from src.i18n import t

FORMULA_CSS = """
<style>
.fv-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 22px 26px 18px 26px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
    margin-bottom: 14px;
    overflow-x: auto;
}
.fv-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
}
.fv-card-title {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #475569;
}
.fv-card-hint {
    font-size: 0.78rem;
    color: #64748B;
}
.fv-math-stage {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.45rem;
    min-height: 92px;
    padding: 16px 8px;
    font-family: 'Cambria Math', 'STIX Two Math', 'Latin Modern Math', Georgia, serif;
    font-size: 1.45rem;
    color: #0F172A;
    line-height: 1.35;
}
.fv-eq-sign {
    font-size: 1.55rem;
    color: #64748B;
    margin: 0 0.35rem;
    font-weight: 400;
}
.fv-rhs-value {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.92rem;
    color: #334155;
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    border-radius: 999px;
    padding: 3px 11px;
    margin-left: 0.45rem;
    white-space: nowrap;
}
.fv-tok {
    position: relative;
    display: inline-flex;
    align-items: baseline;
    font-style: italic;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 8px;
    cursor: help;
    transition: font-size 0.18s ease, background-color 0.18s ease, color 0.18s ease, transform 0.18s ease;
    line-height: 1.1;
    white-space: nowrap;
}
.fv-tok:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
    z-index: 10;
}
.fv-tok-pos {
    color: #047857;
    background: rgba(16, 185, 129, 0.13);
    border-bottom: 2px solid rgba(5, 150, 105, 0.55);
}
.fv-tok-neg {
    color: #B91C1C;
    background: rgba(239, 68, 68, 0.12);
    border-bottom: 2px solid rgba(220, 38, 38, 0.55);
}
.fv-tok-neutral {
    color: #334155;
    background: rgba(148, 163, 184, 0.15);
    border-bottom: 2px solid rgba(100, 116, 139, 0.4);
}
.fv-tok-mono {
    color: #1D4ED8;
    background: rgba(37, 99, 235, 0.10);
    border-bottom: 2px solid rgba(37, 99, 235, 0.45);
}
.fv-tok-response {
    color: #1E40AF;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(99, 102, 241, 0.14));
    border: 1.5px solid rgba(37, 99, 235, 0.35);
    font-style: normal;
    font-weight: 700;
    padding: 3px 10px;
}
.fv-occ-tag {
    font-style: normal;
    font-size: 0.55em;
    vertical-align: sub;
    opacity: 0.72;
    margin-left: 1px;
    font-weight: 600;
}
.fv-frac {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    vertical-align: middle;
    margin: 0 0.18rem;
}
.fv-frac-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0 0.3rem 0.12rem 0.3rem;
    border-bottom: 1.8px solid #334155;
    min-width: 1.2rem;
}
.fv-frac-den {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.16rem 0.3rem 0 0.3rem;
    min-width: 1.2rem;
}
.fv-pow {
    display: inline-flex;
    align-items: flex-start;
}
.fv-sup {
    font-size: 0.72em;
    margin-left: 1px;
    margin-top: -0.45em;
    display: inline-flex;
    align-items: center;
}
.fv-sqrt {
    display: inline-flex;
    align-items: center;
    margin: 0 0.08rem;
}
.fv-sqrt-rad {
    font-size: 1.2em;
    margin-right: 1px;
    color: #334155;
}
.fv-sqrt-body {
    border-top: 1.6px solid #334155;
    padding: 0.08rem 0.22rem 0 0.18rem;
    display: inline-flex;
    align-items: center;
}
.fv-binom {
    display: inline-flex;
    align-items: center;
    margin: 0 0.14rem;
}
.fv-binom-paren {
    font-size: 1.75em;
    font-weight: 300;
    color: #475569;
    line-height: 0.9;
}
.fv-binom-stack {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    font-size: 0.85em;
    padding: 0 0.15rem;
    line-height: 1.1;
}
.fv-fn {
    font-style: normal;
    font-weight: 500;
    color: #334155;
    margin-right: 1px;
}
.fv-op {
    margin: 0 0.2rem;
    color: #475569;
    font-style: normal;
}
.fv-paren {
    color: #475569;
    font-weight: 300;
    font-size: 1.12em;
}
.fv-num {
    font-style: normal;
    color: #1E293B;
}
.fv-legend {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 14px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed #E2E8F0;
    font-size: 0.78rem;
    color: #475569;
}
.fv-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.fv-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}
.fv-breakdown {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 10px;
    margin-top: 10px;
}
.fv-bcard {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 9px 12px;
}
.fv-bcard-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.84rem;
    font-weight: 600;
    color: #0F172A;
}
.fv-bcard-ctx {
    font-size: 0.73rem;
    color: #64748B;
    margin-top: 2px;
    margin-bottom: 6px;
}
.fv-bar-bg {
    height: 7px;
    background: #F1F5F9;
    border-radius: 999px;
    overflow: hidden;
}
.fv-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.2s ease;
}
</style>
"""


def _fmt_num(val: float) -> str:
    if not math.isfinite(val):
        return "—"
    if abs(val) == 0:
        return "0"
    if abs(val) >= 1000 or (0 < abs(val) < 0.001):
        return f"{val:.3e}"
    return f"{val:.4g}"


def _format_sympy_number(node: sp.Expr) -> str:
    if node == sp.pi:
        return '<span class="fv-num">π</span>'
    if node == sp.E:
        return '<span class="fv-num">e</span>'
    if isinstance(node, sp.Integer):
        return f'<span class="fv-num">{int(node)}</span>'
    if isinstance(node, sp.Rational):
        p, q = int(node.p), int(node.q)
        if q == 1:
            return f'<span class="fv-num">{p}</span>'
        sign = '<span class="fv-op">−</span>' if p < 0 else ""
        return (
            f'{sign}<span class="fv-frac">'
            f'<span class="fv-frac-num"><span class="fv-num">{abs(p)}</span></span>'
            f'<span class="fv-frac-den"><span class="fv-num">{q}</span></span>'
            f"</span>"
        )
    if isinstance(node, sp.Float):
        return f'<span class="fv-num">{_fmt_num(float(node))}</span>'
    return f'<span class="fv-num">{html.escape(str(node))}</span>'


def _is_neg_pow(node: sp.Expr) -> bool:
    if isinstance(node, sp.Pow) and node.exp.is_number and node.exp.is_real:
        try:
            return float(node.exp) < 0
        except (TypeError, ValueError):
            return False
    return False


def _is_special_standalone_factor(node: sp.Expr) -> bool:
    """Keep exp(...), log(...), binomial(...) beside leading fraction for textbook layout."""
    if node.func in (sp.exp, sp.log, sp.binomial):
        return True
    # Also keep (x - mu)**(...) or (1 + x^2/nu)**(-...) separated from Gamma/sqrt fractions
    if isinstance(node, sp.Pow) and not node.base.is_number:
        if node.base.is_Add:
            return True
    return False


def _render_occurrence_token(
    occ: OccurrenceAnalysis,
    color_coding: bool,
    lang: str,
    base_rem: float = 1.38,
) -> str:
    meta = occ.meta
    font_rem = round(base_rem * occ.scale, 3)

    if color_coding:
        cls_map = {"pos": "fv-tok-pos", "neg": "fv-tok-neg", "neutral": "fv-tok-neutral"}
        color_cls = cls_map.get(occ.direction, "fv-tok-neutral")
    else:
        color_cls = "fv-tok-mono"

    ctx = meta.context_no if lang == "no" else meta.context_en
    occ_suffix = ""
    if meta.total_occurrences > 1:
        occ_suffix = f'<span class="fv-occ-tag">{meta.occurrence_index}</span>'
        occ_title = f"{meta.display_name} ({t('occurrence_label', lang, idx=meta.occurrence_index)} — {ctx})"
    else:
        occ_title = f"{meta.display_name} ({ctx})"

    dir_key = {"pos": "dir_up", "neg": "dir_down", "neutral": "dir_neutral"}[occ.direction]
    tooltip_lines = [
        occ_title,
        f"{t('col_value', lang)}: {_fmt_num(occ.value)}",
        f"{t('col_partial', lang)}: {_fmt_num(occ.partial_derivative)}",
        f"{t('col_elasticity', lang)}: {_fmt_num(occ.elasticity)}",
        f"{t('col_direction', lang)}: {t(dir_key, lang)}",
        f"{t('col_scale', lang)}: {occ.scale:.2f}×",
    ]
    tooltip = "&#10;".join(html.escape(line, quote=True) for line in tooltip_lines)

    label_safe = html.escape(meta.display_name)
    occ_id_safe = html.escape(meta.occ_id, quote=True)
    return (
        f'<span class="fv-tok {color_cls}" '
        f'data-occ-id="{occ_id_safe}" '
        f'style="font-size:{font_rem}rem;" '
        f'title="{tooltip}">'
        f"{label_safe}{occ_suffix}</span>"
    )


def _node_to_html(
    node: sp.Expr,
    occurrences: Dict[str, OccurrenceAnalysis],
    color_coding: bool,
    lang: str,
    base_rem: float = 1.38,
) -> str:
    """Recursively render a SymPy tagged AST node into HTML."""
    # 1. Tagged occurrence leaf
    if isinstance(node, sp.Symbol):
        if node.name in occurrences:
            return _render_occurrence_token(occurrences[node.name], color_coding, lang, base_rem)
        return f'<span class="fv-num">{html.escape(node.name)}</span>'

    # 2. Numeric constants
    if node.is_number and not node.args:
        return _format_sympy_number(node)
    if isinstance(node, (sp.Integer, sp.Rational, sp.Float)) or node in (sp.pi, sp.E):
        return _format_sympy_number(node)

    # 3. Known mathematical functions
    if node.func == sp.exp:
        inner = _node_to_html(node.args[0], occurrences, color_coding, lang, base_rem)
        return (
            f'<span class="fv-fn">exp</span>'
            f'<span class="fv-paren">(</span>{inner}<span class="fv-paren">)</span>'
        )

    if node.func == sp.log:
        if len(node.args) == 2:
            arg_html = _node_to_html(node.args[0], occurrences, color_coding, lang, base_rem)
            base_html = _node_to_html(node.args[1], occurrences, color_coding, lang, base_rem * 0.72)
            return (
                f'<span class="fv-fn">log<sub>{base_html}</sub></span>'
                f'<span class="fv-paren">(</span>{arg_html}<span class="fv-paren">)</span>'
            )
        inner = _node_to_html(node.args[0], occurrences, color_coding, lang, base_rem)
        return (
            f'<span class="fv-fn">ln</span>'
            f'<span class="fv-paren">(</span>{inner}<span class="fv-paren">)</span>'
        )

    if node.func == sp.gamma:
        inner = _node_to_html(node.args[0], occurrences, color_coding, lang, base_rem)
        return (
            f'<span class="fv-fn">Γ</span>'
            f'<span class="fv-paren">(</span>{inner}<span class="fv-paren">)</span>'
        )

    if node.func == sp.factorial:
        arg = node.args[0]
        inner = _node_to_html(arg, occurrences, color_coding, lang, base_rem)
        if isinstance(arg, (sp.Add, sp.Mul, sp.Pow)):
            inner = f'<span class="fv-paren">(</span>{inner}<span class="fv-paren">)</span>'
        return f'{inner}<span class="fv-fn">!</span>'

    if node.func == sp.binomial:
        top = _node_to_html(node.args[0], occurrences, color_coding, lang, base_rem * 0.92)
        bot = _node_to_html(node.args[1], occurrences, color_coding, lang, base_rem * 0.92)
        return (
            f'<span class="fv-binom">'
            f'<span class="fv-binom-paren">(</span>'
            f'<span class="fv-binom-stack"><span>{top}</span><span>{bot}</span></span>'
            f'<span class="fv-binom-paren">)</span>'
            f"</span>"
        )

    if node.func == sp.Abs:
        inner = _node_to_html(node.args[0], occurrences, color_coding, lang, base_rem)
        return f'<span class="fv-paren">|</span>{inner}<span class="fv-paren">|</span>'

    if node.func in (sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh, sp.erf):
        fn_name = node.func.__name__
        inner = _node_to_html(node.args[0], occurrences, color_coding, lang, base_rem)
        return (
            f'<span class="fv-fn">{html.escape(fn_name)}</span>'
            f'<span class="fv-paren">(</span>{inner}<span class="fv-paren">)</span>'
        )

    # 4. Powers & Roots
    if isinstance(node, sp.Pow):
        base, exp_part = node.args
        # Square root
        if exp_part == sp.Rational(1, 2):
            inner = _node_to_html(base, occurrences, color_coding, lang, base_rem)
            return (
                f'<span class="fv-sqrt">'
                f'<span class="fv-sqrt-rad">√</span>'
                f'<span class="fv-sqrt-body">{inner}</span>'
                f"</span>"
            )
        # Reciprocal square root
        if exp_part == sp.Rational(-1, 2):
            inner = _node_to_html(base, occurrences, color_coding, lang, base_rem)
            denom = (
                f'<span class="fv-sqrt">'
                f'<span class="fv-sqrt-rad">√</span>'
                f'<span class="fv-sqrt-body">{inner}</span>'
                f"</span>"
            )
            return (
                f'<span class="fv-frac">'
                f'<span class="fv-frac-num"><span class="fv-num">1</span></span>'
                f'<span class="fv-frac-den">{denom}</span>'
                f"</span>"
            )
        # Negative integer or rational exponent -> fraction 1 / base^|exp|
        if _is_neg_pow(node):
            pos_exp = -exp_part
            denom_node = base if pos_exp == 1 else sp.Pow(base, pos_exp, evaluate=False)
            denom_html = _node_to_html(denom_node, occurrences, color_coding, lang, base_rem)
            return (
                f'<span class="fv-frac">'
                f'<span class="fv-frac-num"><span class="fv-num">1</span></span>'
                f'<span class="fv-frac-den">{denom_html}</span>'
                f"</span>"
            )

        base_html = _node_to_html(base, occurrences, color_coding, lang, base_rem)
        if isinstance(base, (sp.Add, sp.Mul, sp.Pow)):
            base_html = f'<span class="fv-paren">(</span>{base_html}<span class="fv-paren">)</span>'
        exp_html = _node_to_html(exp_part, occurrences, color_coding, lang, base_rem * 0.85)
        return (
            f'<span class="fv-pow">{base_html}'
            f'<span class="fv-sup">{exp_html}</span></span>'
        )

    # 5. Addition & Subtraction
    if isinstance(node, sp.Add):
        return _render_add(node, occurrences, color_coding, lang, base_rem)

    # 6. Multiplication & Division (Fractions)
    if isinstance(node, sp.Mul):
        return _render_mul(node, occurrences, color_coding, lang, base_rem)

    # Generic fallback
    return f'<span class="fv-num">{html.escape(str(node))}</span>'


def _is_negative_term(term: sp.Expr) -> Tuple[bool, sp.Expr]:
    """Detect whether an additive term has a leading negative sign."""
    if term.is_number and term.is_real:
        try:
            if float(term) < 0:
                return True, -term
        except (TypeError, ValueError):
            pass
    if isinstance(term, sp.Mul):
        for i, arg in enumerate(term.args):
            if arg.is_number and arg.is_real:
                try:
                    if float(arg) < 0:
                        rest = list(term.args)
                        if arg == -1:
                            rest.pop(i)
                        else:
                            rest[i] = -arg
                        if not rest:
                            return True, sp.Integer(1)
                        if len(rest) == 1:
                            return True, rest[0]
                        return True, sp.Mul(*rest, evaluate=False)
                except (TypeError, ValueError):
                    pass
    return False, term


def _render_add(
    node: sp.Add,
    occurrences: Dict[str, OccurrenceAnalysis],
    color_coding: bool,
    lang: str,
    base_rem: float,
) -> str:
    # Place positive terms before negative terms (so x - mu renders as x - mu, 1 - p as 1 - p)
    pos_terms: List[sp.Expr] = []
    neg_terms: List[sp.Expr] = []
    for arg in node.args:
        is_neg, stripped = _is_negative_term(arg)
        if is_neg:
            neg_terms.append(stripped)
        else:
            pos_terms.append(arg)

    ordered: List[Tuple[bool, sp.Expr]] = [(False, t_p) for t_p in pos_terms] + [
        (True, t_n) for t_n in neg_terms
    ]

    parts: List[str] = []
    for idx, (is_neg, term) in enumerate(ordered):
        term_html = _node_to_html(term, occurrences, color_coding, lang, base_rem)
        if idx == 0:
            if is_neg:
                parts.append(f'<span class="fv-op">−</span>{term_html}')
            else:
                parts.append(term_html)
        else:
            op = "−" if is_neg else "+"
            parts.append(f'<span class="fv-op">{op}</span>{term_html}')

    return "".join(parts)


def _render_mul(
    node: sp.Mul,
    occurrences: Dict[str, OccurrenceAnalysis],
    color_coding: bool,
    lang: str,
    base_rem: float,
) -> str:
    sign_neg = False
    numer_factors: List[sp.Expr] = []
    denom_factors: List[sp.Expr] = []
    trailing_factors: List[sp.Expr] = []

    for arg in node.args:
        if arg == -1:
            sign_neg = not sign_neg
            continue
        if isinstance(arg, sp.Integer) and int(arg) < 0:
            sign_neg = not sign_neg
            if int(arg) != -1:
                numer_factors.append(sp.Integer(abs(int(arg))))
            continue
        if isinstance(arg, sp.Rational) and not isinstance(arg, sp.Integer):
            p, q = int(arg.p), int(arg.q)
            if p < 0:
                sign_neg = not sign_neg
                p = abs(p)
            if p != 1:
                numer_factors.append(sp.Integer(p))
            if q != 1:
                denom_factors.append(sp.Integer(q))
            continue
        if _is_neg_pow(arg):
            pos_exp = -arg.exp
            if pos_exp == 1:
                denom_factors.append(arg.base)
            else:
                denom_factors.append(sp.Pow(arg.base, pos_exp, evaluate=False))
            continue
        if _is_special_standalone_factor(arg):
            trailing_factors.append(arg)
            continue
        numer_factors.append(arg)

    # If there are no denominators at all, render everything cleanly as a product
    if not denom_factors:
        all_factors = numer_factors + trailing_factors
        rendered = _join_mul_factors(all_factors, occurrences, color_coding, lang, base_rem)
        prefix = '<span class="fv-op">−</span>' if sign_neg else ""
        return f"{prefix}{rendered}"

    # When there IS a denominator, keep special factors like exp(...) beside the fraction
    # for textbook layout (e.g. 1/(σ√(2π)) · exp(...)), unless numerator is empty and no trailing
    if not numer_factors:
        num_html = '<span class="fv-num">1</span>'
    else:
        num_html = _join_mul_factors(numer_factors, occurrences, color_coding, lang, base_rem)

    den_html = _join_mul_factors(denom_factors, occurrences, color_coding, lang, base_rem)
    frac_html = (
        f'<span class="fv-frac">'
        f'<span class="fv-frac-num">{num_html}</span>'
        f'<span class="fv-frac-den">{den_html}</span>'
        f"</span>"
    )

    prefix = '<span class="fv-op">−</span>' if sign_neg else ""
    if trailing_factors:
        trail_html = _join_mul_factors(trailing_factors, occurrences, color_coding, lang, base_rem)
        return f'{prefix}{frac_html}<span class="fv-op">·</span>{trail_html}'
    return f"{prefix}{frac_html}"


def _join_mul_factors(
    factors: List[sp.Expr],
    occurrences: Dict[str, OccurrenceAnalysis],
    color_coding: bool,
    lang: str,
    base_rem: float,
) -> str:
    if not factors:
        return '<span class="fv-num">1</span>'

    # Combine numeric sqrt factors like sqrt(2) * sqrt(pi) -> √(2π)
    rendered_parts: List[str] = []
    for f in factors:
        h = _node_to_html(f, occurrences, color_coding, lang, base_rem)
        if isinstance(f, sp.Add):
            h = f'<span class="fv-paren">(</span>{h}<span class="fv-paren">)</span>'
        rendered_parts.append(h)

    return '<span class="fv-op">·</span>'.join(rendered_parts)


def render_formula_stage_inner_html(
    parsed: ParsedFormula,
    eval_res: EvaluationResult,
    color_coding: bool = True,
    lang: str = "no",
    base_rem: float = 1.38,
) -> str:
    """Render just the inner LHS = RHS = value markup with data-occ-id hooks for 60 FPS JS updates."""
    lhs_font_rem = round(base_rem * 1.03 * eval_res.response_scale, 3)
    lhs_safe = html.escape(parsed.lhs_display)
    val_str = _fmt_num(eval_res.output_value) if eval_res.valid else "—"

    lhs_tooltip_lines = [
        parsed.lhs_display,
        f"{t('col_value', lang)}: {val_str}",
        f"{t('col_scale', lang)}: {eval_res.response_scale:.2f}×",
    ]
    lhs_tooltip = "&#10;".join(html.escape(line, quote=True) for line in lhs_tooltip_lines)
    lhs_html = (
        f'<span class="fv-tok fv-tok-response" '
        f'data-role="lhs" '
        f'style="font-size:{lhs_font_rem}rem;" '
        f'title="{lhs_tooltip}">{lhs_safe}</span>'
    )

    rhs_html = _node_to_html(
        parsed.expr_tagged,
        occurrences=eval_res.occurrences,
        color_coding=color_coding,
        lang=lang,
        base_rem=base_rem,
    )

    return (
        f"{lhs_html}"
        f'<span class="fv-eq-sign">=</span>'
        f"{rhs_html}"
        f'<span class="fv-rhs-value" data-role="rhs-value">= {html.escape(val_str)}</span>'
    )


def render_formula_card_html(
    parsed: ParsedFormula,
    eval_res: EvaluationResult,
    scaling_mode: str = "sensitivity",
    color_coding: bool = True,
    lang: str = "no",
    dark_mode: bool = False,
) -> str:
    """Render the complete hero formula card HTML with dynamic LHS + RHS visual scaling."""
    stage_inner = render_formula_stage_inner_html(
        parsed=parsed,
        eval_res=eval_res,
        color_coding=color_coding,
        lang=lang,
    )

    hint_key = "formula_card_hint_sens" if scaling_mode == "sensitivity" else "formula_card_hint_val"

    if color_coding and scaling_mode == "sensitivity":
        legend_html = (
            f'<div class="fv-legend">'
            f'<span class="fv-legend-item"><span class="fv-dot" style="background:#059669;"></span>{t("legend_pos", lang)}</span>'
            f'<span class="fv-legend-item"><span class="fv-dot" style="background:#DC2626;"></span>{t("legend_neg", lang)}</span>'
            f'<span class="fv-legend-item"><span class="fv-dot" style="background:#64748B;"></span>{t("legend_neutral", lang)}</span>'
            f'<span class="fv-legend-item"><span class="fv-dot" style="background:#2563EB;"></span>{t("legend_response", lang)}</span>'
            f"</div>"
        )
    else:
        legend_html = (
            f'<div class="fv-legend">'
            f'<span class="fv-legend-item"><span class="fv-dot" style="background:#2563EB;"></span>{t("legend_response", lang)}</span>'
            f"</div>"
        )

    theme_cls = " fv-dark" if dark_mode else ""
    return (
        f"{FORMULA_CSS}"
        f'<div class="fv-card{theme_cls}">'
        f'<div class="fv-card-header">'
        f'<span class="fv-card-title">{html.escape(t("formula_card_title", lang))}</span>'
        f'<span class="fv-card-hint">{html.escape(t(hint_key, lang))}</span>'
        f"</div>"
        f'<div class="fv-math-stage">'
        f"{stage_inner}"
        f"</div>"
        f"{legend_html}"
        f"</div>"
    )


def render_breakdown_html(
    eval_res: EvaluationResult,
    color_coding: bool = True,
    lang: str = "no",
) -> str:
    """Render compact visual proportion cards for every variable occurrence in the formula."""
    items = list(eval_res.occurrences.values())
    if not items:
        return f'<p style="color:#64748B;font-size:0.85rem;">{html.escape(t("breakdown_empty", lang))}</p>'

    cards: List[str] = []
    for occ in items:
        meta = occ.meta
        ctx = meta.context_no if lang == "no" else meta.context_en
        idx_tag = f" #{meta.occurrence_index}" if meta.total_occurrences > 1 else ""
        pct = max(2.0, min(100.0, occ.relative_share * 100.0))

        if color_coding:
            bar_color = {"pos": "#10B981", "neg": "#EF4444", "neutral": "#94A3B8"}[occ.direction]
            sign_arrow = {"pos": "↑", "neg": "↓", "neutral": "→"}[occ.direction]
        else:
            bar_color = "#2563EB"
            sign_arrow = "•"

        cards.append(
            f'<div class="fv-bcard">'
            f'<div class="fv-bcard-top">'
            f"<span>{html.escape(meta.display_name)}{idx_tag} = {html.escape(_fmt_num(occ.value))}</span>"
            f'<span style="color:{bar_color};">{sign_arrow} {occ.scale:.2f}×</span>'
            f"</div>"
            f'<div class="fv-bcard-ctx">{html.escape(ctx)} · E = {html.escape(_fmt_num(occ.elasticity))}</div>'
            f'<div class="fv-bar-bg"><div class="fv-bar-fill" style="width:{pct:.1f}%;background:{bar_color};"></div></div>'
            f"</div>"
        )

    return f'<div class="fv-breakdown">{"".join(cards)}</div>'
