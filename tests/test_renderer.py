"""Tests for dynamic HTML/CSS formula rendering and breakdown cards."""

from src.distributions import DISTRIBUTIONS, get_distribution_parsed
from src.examples import EXAMPLE_FORMULAS
from src.formula_engine import evaluate_formula_point, parse_custom_formula
from src.formula_renderer import render_breakdown_html, render_formula_card_html


def test_render_normal_distribution_html_both_languages():
    spec = DISTRIBUTIONS["normal"]
    parsed = get_distribution_parsed(spec)
    eval_res = evaluate_formula_point(parsed, {"mu": 0.0, "sigma": 0.5, "x": 1.0})

    for lang in ("no", "en"):
        card = render_formula_card_html(parsed, eval_res, scaling_mode="sensitivity", color_coding=True, lang=lang)
        assert "fv-card" in card
        assert "fv-tok-response" in card
        assert "fv-tok-pos" in card
        assert "fv-tok-neg" in card
        assert "σ" in card
        assert "font-size:" in card

        breakdown = render_breakdown_html(eval_res, color_coding=True, lang=lang)
        assert "fv-breakdown" in breakdown
        assert "fv-bcard" in breakdown


def test_render_all_example_formulas():
    for ex in EXAMPLE_FORMULAS:
        parsed = parse_custom_formula(ex.formula)
        values = {}
        for var in parsed.variables:
            disp = parsed.display_names[var]
            values[var] = ex.defaults.get(disp, parsed.default_values[var])

        res = evaluate_formula_point(parsed, values, scaling_mode="sensitivity")
        assert res.valid, f"Example {ex.key} failed to evaluate"

        html_out = render_formula_card_html(parsed, res, scaling_mode="sensitivity", color_coding=True, lang="no")
        assert "fv-math-stage" in html_out
        assert parsed.lhs_display.replace("&", "&amp;") in html_out
