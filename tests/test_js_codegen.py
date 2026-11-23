"""Tests for SymPy -> JavaScript compilation and 60 FPS live studio HTML generation."""

import re
import shutil
import subprocess
import tempfile

import pytest

from src.distributions import DISTRIBUTIONS, get_distribution_parsed
from src.examples import EXAMPLE_FORMULAS
from src.formula_engine import evaluate_formula_point, parse_custom_formula
from src.js_codegen import compile_parsed_to_js_spec
from src.live_widget import render_live_studio_html


@pytest.mark.parametrize("dist_key", list(DISTRIBUTIONS.keys()))
def test_distribution_js_codegen_and_live_studio(dist_key: str):
    spec = DISTRIBUTIONS[dist_key]
    parsed = get_distribution_parsed(spec)
    js_spec = compile_parsed_to_js_spec(parsed)

    assert js_spec["eval_js"] is not None
    assert js_spec["tagged_eval_js"] is not None
    assert len(js_spec["elasticity_js"]) == len(parsed.occ_ids_ordered)

    vals = {p.name: float(p.default) for p in spec.params}
    vals[spec.indep_var] = float(spec.default_focus)
    eval_res = evaluate_formula_point(parsed, vals)
    assert eval_res.valid

    sliders = [
        {
            "name": p.name,
            "label": p.label_no,
            "min": float(p.min_val),
            "max": float(p.max_val),
            "step": float(p.step),
            "value": float(p.default),
            "is_int": p.is_int,
            "is_focus": False,
        }
        for p in spec.params
    ]
    sliders.append(
        {
            "name": spec.indep_var,
            "label": f"Fokuspunkt ({spec.indep_var}₀)",
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
            "value": float(spec.default_focus),
            "is_int": spec.kind == "discrete",
            "is_focus": True,
        }
    )

    for dark_mode in (False, True):
        studio_html = render_live_studio_html(
            parsed=parsed,
            initial_eval=eval_res,
            sliders=sliders,
            mode_kind=parsed.output_type,
            dist_key=dist_key,
            sweep_var=spec.indep_var,
            sweep_options=[{"name": spec.indep_var, "label": spec.indep_var}],
            title=spec.name_no,
            insight_text=spec.insight_no,
            lang="no",
            dark_mode=dark_mode,
        )
        assert "LIVE 60 FPS" in studio_html
        assert 'id="liveChart"' in studio_html
        assert ('class="dark"' in studio_html) == dark_mode
        assert 'data-role="lhs"' in studio_html


@pytest.mark.parametrize("example", EXAMPLE_FORMULAS, ids=lambda e: e.key)
def test_example_formula_js_codegen_and_live_studio(example):
    parsed = parse_custom_formula(example.formula)
    js_spec = compile_parsed_to_js_spec(parsed)
    assert js_spec["eval_js"] is not None
    assert js_spec["tagged_eval_js"] is not None

    vals = {
        v: example.defaults.get(parsed.display_names[v], parsed.default_values[v])
        for v in parsed.variables
    }
    eval_res = evaluate_formula_point(parsed, vals)
    assert eval_res.valid

    sliders = [
        {
            "name": v,
            "label": parsed.display_names[v],
            "min": 0.1,
            "max": 100.0,
            "step": 0.1,
            "value": vals[v],
            "is_int": False,
            "is_focus": False,
        }
        for v in parsed.variables
    ]
    studio_html = render_live_studio_html(
        parsed=parsed,
        initial_eval=eval_res,
        sliders=sliders,
        mode_kind=parsed.output_type,
        dist_key=None,
        sweep_var=parsed.variables[0],
        sweep_options=[{"name": v, "label": parsed.display_names[v]} for v in parsed.variables],
        title=example.title_no,
        lang="no",
        dark_mode=True,
    )
    assert 'id="formulaStage"' in studio_html
    assert 'id="liveChart"' in studio_html


def test_generated_studio_javascript_syntax_is_valid():
    """Validate the embedded JavaScript bundle with `node --check` when Node.js is available."""
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("node not installed")

    spec = DISTRIBUTIONS["normal"]
    parsed = get_distribution_parsed(spec)
    eval_res = evaluate_formula_point(parsed, {"mu": 0.0, "sigma": 1.0, "x": 1.2})
    studio_html = render_live_studio_html(
        parsed=parsed,
        initial_eval=eval_res,
        sliders=[
            {"name": "mu", "label": "μ", "min": -5, "max": 5, "step": 0.1, "value": 0.0, "is_int": False, "is_focus": False},
            {"name": "sigma", "label": "σ", "min": 0.25, "max": 4, "step": 0.05, "value": 1.0, "is_int": False, "is_focus": False},
            {"name": "x", "label": "x₀", "min": -6, "max": 6, "step": 0.05, "value": 1.2, "is_int": False, "is_focus": True},
        ],
        mode_kind="continuous_dist",
        dist_key="normal",
        sweep_var="x",
        sweep_options=[{"name": "x", "label": "x"}],
        title="Normalfordeling",
        lang="no",
        dark_mode=True,
    )
    m = re.search(r"<script>([\s\S]*?)</script>", studio_html)
    assert m is not None
    js_code = m.group(1)
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=True) as tmp:
        tmp.write(js_code)
        tmp.flush()
        proc = subprocess.run([node_bin, "--check", tmp.name], capture_output=True, text=True)
        assert proc.returncode == 0, f"JS syntax error:\n{proc.stderr}"
