"""Unit tests for custom formula parsing, output inference, and per-occurrence scaling."""

import math

import numpy as np
import pytest

from src.formula_engine import evaluate_curve, evaluate_formula_point, parse_custom_formula


def test_parse_tfidf_formula():
    parsed = parse_custom_formula("tfidf(t, d) = tf(t, d) * log(N/df(t))")
    assert parsed.lhs_display == "tfidf(t, d)"
    assert parsed.output_type == "scalar"
    assert parsed.independent_var is None

    disp_to_var = {v: k for k, v in parsed.display_names.items()}
    assert "tf(t, d)" in disp_to_var
    assert "df(t)" in disp_to_var
    assert "N" in disp_to_var

    values = {
        disp_to_var["tf(t, d)"]: 4.0,
        disp_to_var["N"]: 100.0,
        disp_to_var["df(t)"]: 10.0,
    }
    res = evaluate_formula_point(parsed, values, scaling_mode="sensitivity")
    assert res.valid
    expected = 4.0 * math.log(100.0 / 10.0)
    assert math.isclose(res.output_value, expected, rel_tol=1e-6)

    # Check effect directions: tf and N increase TF-IDF, df decreases TF-IDF
    by_disp = {occ.meta.display_name: occ for occ in res.occurrences.values()}
    assert by_disp["tf(t, d)"].direction == "pos"
    assert by_disp["N"].direction == "pos"
    assert by_disp["df(t)"].direction == "neg"
    assert math.isclose(by_disp["tf(t, d)"].elasticity, 1.0, rel_tol=1e-5)


def test_parse_inverse_scalar_b_equals_1_over_a():
    parsed = parse_custom_formula("b = 1/a")
    assert parsed.lhs_display == "b"
    assert parsed.output_type == "scalar"
    assert parsed.variables == ["a"]

    base = evaluate_formula_point(parsed, {"a": 1.0}, scaling_mode="sensitivity")
    base_partials = {oid: o.partial_derivative for oid, o in base.occurrences.items()}

    # Sensitivity mode: smaller 'a' -> bigger response 'b' AND larger local sensitivity of 'a'
    small_a_sens = evaluate_formula_point(
        parsed,
        {"a": 0.25},
        scaling_mode="sensitivity",
        baseline_output=base.output_value,
        baseline_values={"a": 1.0},
        baseline_partials=base_partials,
    )
    large_a_sens = evaluate_formula_point(
        parsed,
        {"a": 4.0},
        scaling_mode="sensitivity",
        baseline_output=base.output_value,
        baseline_values={"a": 1.0},
        baseline_partials=base_partials,
    )
    assert math.isclose(small_a_sens.output_value, 4.0, rel_tol=1e-6)
    assert math.isclose(large_a_sens.output_value, 0.25, rel_tol=1e-6)
    assert small_a_sens.response_scale > large_a_sens.response_scale

    occ_small = next(iter(small_a_sens.occurrences.values()))
    occ_large = next(iter(large_a_sens.occurrences.values()))
    assert occ_small.direction == "neg"
    assert occ_small.scale > occ_large.scale

    # Value mode: smaller 'a' -> smaller visual 'a', larger visual 'b'
    small_a_val = evaluate_formula_point(
        parsed,
        {"a": 0.25},
        scaling_mode="value",
        baseline_output=base.output_value,
        baseline_values={"a": 1.0},
    )
    large_a_val = evaluate_formula_point(
        parsed,
        {"a": 4.0},
        scaling_mode="value",
        baseline_output=base.output_value,
        baseline_values={"a": 1.0},
    )
    val_occ_small = next(iter(small_a_val.occurrences.values()))
    val_occ_large = next(iter(large_a_val.occurrences.values()))
    assert val_occ_small.scale < val_occ_large.scale
    assert small_a_val.response_scale > large_a_val.response_scale


def test_normal_two_sigma_dominance_crossover():
    """Verify the two occurrences of sigma in the Normal PDF swap visual dominance across sigma = |x - mu|."""
    parsed = parse_custom_formula(
        "f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-((x - mu)^2) / (2 * sigma^2))"
    )
    assert parsed.output_type == "continuous_func"
    assert parsed.independent_var == "x"

    sigma_occs = [m for m in parsed.occurrences.values() if m.var_name == "sigma"]
    assert len(sigma_occs) == 2

    # Identify leading (in denominator, NOT in exponent) vs exponent occurrence
    leading_id = next(m.occ_id for m in sigma_occs if "exponent" not in m.context_en)
    exp_id = next(m.occ_id for m in sigma_occs if "exponent" in m.context_en)

    # Case 1: small sigma (0.5 < |x - mu| = 1.0) -> exponent sigma has stronger effect!
    res_small = evaluate_formula_point(parsed, {"x": 1.0, "mu": 0.0, "sigma": 0.5})
    assert res_small.valid
    assert res_small.occurrences[leading_id].direction == "neg"
    assert res_small.occurrences[exp_id].direction == "pos"
    assert abs(res_small.occurrences[exp_id].elasticity) > abs(
        res_small.occurrences[leading_id].elasticity
    )
    assert res_small.occurrences[exp_id].scale > res_small.occurrences[leading_id].scale

    # Case 2: exact crossover (sigma = |x - mu| = 1.0) -> equal magnitude!
    res_eq = evaluate_formula_point(parsed, {"x": 1.0, "mu": 0.0, "sigma": 1.0})
    assert math.isclose(
        abs(res_eq.occurrences[exp_id].elasticity),
        abs(res_eq.occurrences[leading_id].elasticity),
        rel_tol=1e-5,
    )
    assert math.isclose(
        res_eq.occurrences[exp_id].scale,
        res_eq.occurrences[leading_id].scale,
        rel_tol=1e-5,
    )

    # Case 3: large sigma (2.0 > |x - mu| = 1.0) -> leading sigma has stronger effect!
    res_large = evaluate_formula_point(parsed, {"x": 1.0, "mu": 0.0, "sigma": 2.0})
    assert abs(res_large.occurrences[leading_id].elasticity) > abs(
        res_large.occurrences[exp_id].elasticity
    )
    assert res_large.occurrences[leading_id].scale > res_large.occurrences[exp_id].scale


def test_parse_bayes_and_discrete_and_curve_eval():
    bayes = parse_custom_formula("P(A|B) = P(B|A) * P(A) / P(B)")
    assert bayes.output_type == "scalar"
    assert len(bayes.variables) == 3

    binom = parse_custom_formula("p(k) = binomial(n, k) * p^k * (1 - p)^(n - k)")
    assert binom.output_type == "discrete_func"
    assert binom.independent_var == "k"

    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = evaluate_curve(binom, "k", grid, {"k": 2.0, "n": 4.0, "p": 0.5})
    assert math.isclose(float(np.sum(y)), 1.0, rel_tol=1e-5)


def test_invalid_formulas_raise_value_error():
    with pytest.raises(ValueError):
        parse_custom_formula("   ")
    with pytest.raises(ValueError):
        parse_custom_formula("y = __import__('os').system('ls')")
