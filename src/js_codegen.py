"""Compile SymPy expressions and per-occurrence elasticities into fast browser JavaScript."""

from __future__ import annotations

from typing import Dict, List, Optional

import sympy as sp

from src.formula_engine import ParsedFormula, _reevaluate_expr


def sympy_to_js(expr: sp.Expr, var_container: str = "v") -> Optional[str]:
    """
    Convert a SymPy expression into a self-contained JavaScript expression string
    that reads variables from `var_container["name"]` and uses `_M` for special math functions.
    Returns None if the expression contains unsupported symbolic constructs (e.g., Derivative).
    """
    if expr.has(sp.Derivative) or expr.has(sp.Integral):
        return None
    try:
        return _emit(expr, var_container)
    except (ValueError, TypeError):
        return None


def _emit(node: sp.Expr, vc: str) -> str:
    if isinstance(node, sp.Symbol):
        safe_key = node.name.replace("\\", "\\\\").replace('"', '\\"')
        return f'{vc}["{safe_key}"]'

    if node == sp.pi:
        return "Math.PI"
    if node == sp.E:
        return "Math.E"
    if node == sp.oo:
        return "Infinity"
    if node == -sp.oo:
        return "(-Infinity)"
    if node == sp.nan or node == sp.zoo:
        return "NaN"

    if isinstance(node, sp.Integer):
        return str(int(node))

    if isinstance(node, sp.Rational):
        return f"({int(node.p)}/{int(node.q)})"

    if isinstance(node, sp.Float):
        return repr(float(node))

    if isinstance(node, sp.Add):
        parts = [_emit(a, vc) for a in node.args]
        return "(" + " + ".join(parts) + ")"

    if isinstance(node, sp.Mul):
        parts = [_emit(a, vc) for a in node.args]
        return "(" + " * ".join(parts) + ")"

    if isinstance(node, sp.Pow):
        base, exp_part = node.args
        b_js = _emit(base, vc)
        if exp_part == sp.Integer(2):
            return f"((({b_js}) * ({b_js})))"
        if exp_part == sp.Integer(-1):
            return f"(1.0 / ({b_js}))"
        if exp_part == sp.Rational(1, 2):
            return f"Math.sqrt({b_js})"
        if exp_part == sp.Rational(-1, 2):
            return f"(1.0 / Math.sqrt({b_js}))"
        e_js = _emit(exp_part, vc)
        return f"Math.pow({b_js}, {e_js})"

    func = node.func
    if func == sp.exp:
        return f"Math.exp({_emit(node.args[0], vc)})"
    if func == sp.log:
        if len(node.args) == 2:
            return f"(Math.log({_emit(node.args[0], vc)}) / Math.log({_emit(node.args[1], vc)}))"
        return f"Math.log({_emit(node.args[0], vc)})"
    if func == sp.sin:
        return f"Math.sin({_emit(node.args[0], vc)})"
    if func == sp.cos:
        return f"Math.cos({_emit(node.args[0], vc)})"
    if func == sp.tan:
        return f"Math.tan({_emit(node.args[0], vc)})"
    if func == sp.asin:
        return f"Math.asin({_emit(node.args[0], vc)})"
    if func == sp.acos:
        return f"Math.acos({_emit(node.args[0], vc)})"
    if func == sp.atan:
        return f"Math.atan({_emit(node.args[0], vc)})"
    if func == sp.sinh:
        return f"Math.sinh({_emit(node.args[0], vc)})"
    if func == sp.cosh:
        return f"Math.cosh({_emit(node.args[0], vc)})"
    if func == sp.tanh:
        return f"Math.tanh({_emit(node.args[0], vc)})"
    if func == sp.Abs:
        return f"Math.abs({_emit(node.args[0], vc)})"
    if func == sp.ceiling:
        return f"Math.ceil({_emit(node.args[0], vc)})"
    if func == sp.floor:
        return f"Math.floor({_emit(node.args[0], vc)})"
    if func == sp.Max:
        args = ", ".join(_emit(a, vc) for a in node.args)
        return f"Math.max({args})"
    if func == sp.Min:
        args = ", ".join(_emit(a, vc) for a in node.args)
        return f"Math.min({args})"
    if func == sp.gamma:
        return f"_M.gamma({_emit(node.args[0], vc)})"
    if func == sp.factorial:
        return f"_M.gamma(({_emit(node.args[0], vc)}) + 1.0)"
    if func == sp.binomial:
        return f"_M.binom({_emit(node.args[0], vc)}, {_emit(node.args[1], vc)})"
    if func == sp.erf:
        return f"_M.erf({_emit(node.args[0], vc)})"
    if func == sp.polygamma:
        if node.args[0] == sp.Integer(0):
            return f"_M.digamma({_emit(node.args[1], vc)})"
        raise ValueError("Only polygamma order 0 (digamma) is supported in JS runtime.")

    raise ValueError(f"Unsupported SymPy node for JS emission: {func}")


def compile_parsed_to_js_spec(parsed: ParsedFormula) -> Dict[str, object]:
    """
    Generate JavaScript expression strings for:
      - `eval_js`: evaluates `parsed.expr` using original variable names (`v["sigma"]`)
      - `tagged_eval_js`: evaluates `parsed.expr_tagged` using occurrence IDs (`v["__occ_0"]`)
      - `elasticity_js`: per-occurrence symbolically cancelled elasticity expressions
      - `partial_js`: per-occurrence symbolic partial derivatives
    """
    eval_js = sympy_to_js(parsed.expr, var_container="v")
    tagged_eval_js = sympy_to_js(parsed.expr_tagged, var_container="v")

    diff_base_expr = _reevaluate_expr(parsed.expr_tagged).replace(
        sp.binomial,
        lambda n, k: sp.gamma(n + 1) / (sp.gamma(k + 1) * sp.gamma(n - k + 1)),
    ).replace(sp.factorial, lambda k: sp.gamma(k + 1))

    elasticity_js: Dict[str, Optional[str]] = {}
    partial_js: Dict[str, Optional[str]] = {}

    for oid in parsed.occ_ids_ordered:
        sym = sp.Symbol(oid, real=True)
        try:
            d_expr = sp.diff(diff_base_expr, sym)
            partial_js[oid] = sympy_to_js(d_expr, var_container="v")
            e_expr = sp.cancel(sym * d_expr / diff_base_expr)
            elasticity_js[oid] = sympy_to_js(e_expr, var_container="v")
        except Exception:
            partial_js[oid] = None
            elasticity_js[oid] = None

    return {
        "eval_js": eval_js,
        "tagged_eval_js": tagged_eval_js,
        "elasticity_js": elasticity_js,
        "partial_js": partial_js,
    }
