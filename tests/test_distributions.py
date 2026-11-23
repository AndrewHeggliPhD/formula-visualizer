"""Unit tests for all 10 built-in statistical probability distributions."""

import math

import numpy as np
import pytest

from src.distributions import (
    DISTRIBUTIONS,
    compute_distribution_stats,
    evaluate_distribution_curve,
    get_distribution_parsed,
)
from src.formula_engine import evaluate_formula_point


@pytest.mark.parametrize("dist_key", list(DISTRIBUTIONS.keys()))
def test_distribution_evaluates_and_matches_scipy(dist_key: str):
    spec = DISTRIBUTIONS[dist_key]
    parsed = get_distribution_parsed(spec)

    param_vals = {p.name: float(p.default) for p in spec.params}
    full_vals = dict(param_vals)
    full_vals[spec.indep_var] = float(spec.default_focus)

    # 1. Symbolic evaluation matches SciPy PDF/PMF at default focus point
    eval_res = evaluate_formula_point(parsed, full_vals, scaling_mode="sensitivity")
    assert eval_res.valid, f"{dist_key} failed evaluation: {eval_res.error_message}"

    scipy_obj = spec.scipy_dist(param_vals)
    expected = (
        float(scipy_obj.pmf(spec.default_focus))
        if spec.kind == "discrete"
        else float(scipy_obj.pdf(spec.default_focus))
    )
    assert math.isclose(eval_res.output_value, expected, rel_tol=1e-4), (
        f"{dist_key}: got {eval_res.output_value}, expected {expected}"
    )

    # 2. Value mode also evaluates cleanly
    eval_val = evaluate_formula_point(parsed, full_vals, scaling_mode="value")
    assert eval_val.valid

    # 3. Both PDF/PMF and CDF curves evaluate cleanly
    for view_type in ("pdf", "cdf"):
        xg, yg = evaluate_distribution_curve(spec, param_vals, view_type=view_type)
        assert len(xg) > 5
        assert len(xg) == len(yg)
        assert np.all(np.isfinite(yg))

    # 4. Moments compute without error
    st_dict = compute_distribution_stats(spec, param_vals)
    assert set(st_dict.keys()) == {"mean", "var", "std"}
    assert math.isfinite(st_dict["mean"])
