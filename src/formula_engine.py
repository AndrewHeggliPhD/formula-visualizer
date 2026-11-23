"""Symbolic formula parsing, output-type inference, and per-occurrence sensitivity analysis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.special as sp_special
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Known mathematical function names that should NOT be treated as compound symbol calls
KNOWN_FUNCTIONS = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "ln",
    "log10",
    "log2",
    "sqrt",
    "abs",
    "Abs",
    "gamma",
    "factorial",
    "binomial",
    "erf",
    "ceil",
    "floor",
    "Max",
    "Min",
}

GREEK_MAP = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "lam": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Sigma": "Σ",
    "Delta": "Δ",
    "Gamma": "Γ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Omega": "Ω",
}

# Safe character whitelist for custom formulas (prevents arbitrary code injection)
_ALLOWED_RE = re.compile(r"^[A-Za-z0-9_+\-*/^().,|= \t\u0370-\u03FF]+$")

_GREEK_TO_ASCII = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "θ": "theta",
    "λ": "lam",
    "μ": "mu",
    "ν": "nu",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}

_NUMPY_SCIPY_MODULES: List[Any] = [
    {
        "gamma": sp_special.gamma,
        "factorial": lambda z: sp_special.gamma(np.asarray(z, dtype=float) + 1.0),
        "binomial": lambda n, k: sp_special.comb(n, k, exact=False),
        "polygamma": lambda m, z: sp_special.polygamma(int(m), z),
        "digamma": sp_special.digamma,
        "erf": sp_special.erf,
        "Abs": np.abs,
        "Max": np.maximum,
        "Min": np.minimum,
    },
    "numpy",
]


@dataclass
class OccurrenceMeta:
    """Metadata for a single syntactic occurrence of a variable in an expression."""

    occ_id: str  # e.g. "__occ_0"
    var_name: str  # internal SymPy symbol name, e.g. "sigma" or "_csym_0"
    display_name: str  # human-readable label, e.g. "σ" or "tf(t, d)"
    occurrence_index: int  # 1-based index among occurrences of the same variable
    total_occurrences: int  # total times this variable appears in the expression
    context_no: str  # Norwegian description of syntactic position
    context_en: str  # English description of syntactic position
    is_independent: bool = False  # True if this is the x/k axis variable


@dataclass
class OccurrenceAnalysis:
    """Computed numerical sensitivity/scaling result for one symbol occurrence."""

    meta: OccurrenceMeta
    value: float
    partial_derivative: float
    elasticity: float
    direction: str  # "pos", "neg", "neutral"
    scale: float  # visual CSS scale multiplier (e.g. 0.62 to 2.05)
    relative_share: float  # share of total effect in [0, 1]


@dataclass
class ParsedFormula:
    """Parsed representation of a mathematical formula ready for analysis and rendering."""

    raw_formula: str
    lhs_display: str
    expr: sp.Expr
    expr_tagged: sp.Expr
    variables: List[str]  # ordered internal symbol names
    display_names: Dict[str, str]  # internal name -> pretty display name
    occurrences: Dict[str, OccurrenceMeta]  # occ_id -> OccurrenceMeta
    output_type: str  # "continuous_dist", "discrete_dist", "continuous_func", "discrete_func", "scalar"
    independent_var: Optional[str]  # internal name of x/k if function/distribution, else None
    eval_fn: Callable[..., Any]
    tagged_eval_fn: Callable[..., Any]
    occ_ids_ordered: List[str]
    elasticity_fns: Dict[str, Optional[Callable[..., Any]]] = field(default_factory=dict)
    partial_fns: Dict[str, Optional[Callable[..., Any]]] = field(default_factory=dict)
    default_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation & dynamic scaling snapshot at a specific parameter setting."""

    output_value: float
    response_scale: float
    response_direction: str
    occurrences: Dict[str, OccurrenceAnalysis]
    valid: bool = True
    error_message: Optional[str] = None


def pretty_symbol_name(name: str) -> str:
    """Convert ASCII Greek names like 'sigma' to 'σ' while leaving normal names intact."""
    if name in GREEK_MAP:
        return GREEK_MAP[name]
    # Handle subscripted Greek like sigma_1 or mu_0
    if "_" in name:
        base, rest = name.split("_", 1)
        if base in GREEK_MAP:
            return f"{GREEK_MAP[base]}_{rest}"
    return name


def _normalize_greek_input(text: str) -> str:
    """Replace Unicode Greek letters in user input with safe ASCII equivalents."""
    for ch, ascii_name in _GREEK_TO_ASCII.items():
        text = text.replace(ch, ascii_name)
    return text


def _extract_compound_symbols(rhs: str) -> Tuple[str, Dict[str, str]]:
    """
    Detect compound function-like inputs on the RHS such as `tf(t, d)`, `df(t)`, `P(B|A)`, `P(A)`.
    Replaces each unique non-math function call with a placeholder symbol `_csym_N` and returns
    the transformed RHS plus a mapping `{placeholder: original_label}`.
    """
    mapping: Dict[str, str] = {}
    reverse: Dict[str, str] = {}
    result: List[str] = []
    i = 0
    n = len(rhs)

    while i < n:
        # Only match identifiers at a word boundary (never mid-identifier!)
        at_word_start = (i == 0) or not (rhs[i - 1].isalnum() or rhs[i - 1] == "_")
        m = re.match(r"([A-Za-z][A-Za-z0-9_]*)\s*\(", rhs[i:]) if at_word_start else None
        if m:
            ident = m.group(1)
            open_paren_idx = i + m.end() - 1
            if ident in KNOWN_FUNCTIONS:
                # Preserve known math function call intact and continue parsing inside its arguments
                result.append(rhs[i : open_paren_idx + 1])
                i = open_paren_idx + 1
                continue

            # Find matching closing parenthesis for compound symbol like tf(t, d) or P(B|A)
            depth = 0
            j = open_paren_idx
            while j < n:
                if rhs[j] == "(":
                    depth += 1
                elif rhs[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1

            if depth == 0:
                full_token = rhs[i : j + 1]
                # Normalize whitespace cleanly for display
                clean_label = re.sub(r"\s+", " ", full_token).replace("( ", "(").replace(" )", ")")
                if clean_label not in reverse:
                    sym_name = f"_csym_{len(mapping)}"
                    mapping[sym_name] = clean_label
                    reverse[clean_label] = sym_name
                result.append(reverse[clean_label])
                i = j + 1
                continue

        result.append(rhs[i])
        i += 1

    return "".join(result), mapping


def _preprocess_rhs(rhs: str) -> Tuple[str, Dict[str, str]]:
    """Normalize user RHS syntax into SymPy-parseable form."""
    text = _normalize_greek_input(rhs.strip())
    # Replace unicode minus / multiplication signs
    text = text.replace("−", "-").replace("×", "*").replace("·", "*")
    # Extract compound inputs like tf(t, d), df(t), P(B|A)
    text, compound_map = _extract_compound_symbols(text)

    # Convert log10(...) and log2(...) before general parsing
    text = re.sub(r"\bln\s*\(", "log(", text)
    # Handle Euler's constant e^(...) or e**(...)
    text = re.sub(r"\be\s*(\^|\*\*)", r"E\1", text)
    # Replace reserved Python keyword 'lambda' as a standalone identifier
    text = re.sub(r"\blambda\b", "lam", text)

    return text, compound_map


def _convert_base_logs(expr: sp.Expr) -> sp.Expr:
    """Already handled via local_dict, kept as safety hook."""
    return expr


def _build_local_dict(processed_rhs: str, compound_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Construct a safe SymPy local_dict for parsing mathematical expressions.
    Explicitly registers every non-function identifier (such as N, S, Q, O, I) as a real
    SymPy Symbol so SymPy never shadows user variables with built-in functions like sympy.N.
    """
    fn_map: Dict[str, Any] = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "log10": lambda u: sp.log(u, 10),
        "log2": lambda u: sp.log(u, 2),
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "gamma": sp.gamma,
        "factorial": sp.factorial,
        "binomial": sp.binomial,
        "erf": sp.erf,
        "ceil": sp.ceiling,
        "floor": sp.floor,
        "Max": sp.Max,
        "Min": sp.Min,
    }
    local_dict: Dict[str, Any] = {
        "pi": sp.pi,
        "E": sp.E,
    }
    for sym_name in compound_map:
        local_dict[sym_name] = sp.Symbol(sym_name, real=True)

    # Identify all identifiers in processed_rhs and bind calls vs variables explicitly
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(\s*\()?", processed_rhs):
        ident = m.group(1)
        is_call = m.group(2) is not None
        if is_call and ident in fn_map:
            local_dict[ident] = fn_map[ident]
        elif ident not in local_dict:
            local_dict[ident] = sp.Symbol(ident, real=True)

    return local_dict


def _infer_context_labels(path: List[str]) -> Tuple[str, str]:
    """Produce human-readable Norwegian and English location hints from AST ancestry."""
    if not path:
        return "hovedledd", "main term"

    if "exp" in path:
        if "denom" in path:
            return "nevner i eksponenten", "denominator in exponent"
        return "i eksponenten", "in exponent"
    if "log" in path:
        if "denom" in path:
            return "nevner i log(...)", "denominator in log(...)"
        return "inne i log(...)", "inside log(...)"
    if "gamma" in path:
        if "denom" in path:
            return "i Γ(...) i nevneren", "in Γ(...) denominator"
        return "inne i Γ(...)", "inside Γ(...)"
    if "binomial" in path:
        return "i binomialkoeffisient", "in binomial coefficient"
    if "sqrt" in path:
        if "denom" in path:
            return "i √(...) i nevneren", "in √(...) denominator"
        return "inne i √(...)", "inside √(...)"
    if "power_exp" in path:
        return "som eksponent", "as exponent"
    if "denom" in path:
        return "i nevneren", "in denominator"
    if "numer" in path:
        return "i telleren", "in numerator"
    if "factor" in path:
        return "faktor", "factor"
    return "i uttrykket", "in expression"


def _is_negative_power(node: sp.Expr) -> bool:
    if isinstance(node, sp.Pow):
        exp_val = node.exp
        if exp_val.is_number and exp_val.is_real:
            try:
                return float(exp_val) < 0
            except (TypeError, ValueError):
                return False
    return False


def tag_expression_occurrences(
    expr: sp.Expr,
    display_names: Dict[str, str],
    independent_var: Optional[str] = None,
) -> Tuple[sp.Expr, Dict[str, OccurrenceMeta], List[str]]:
    """
    Walk the SymPy AST and replace every individual symbol occurrence with a unique
    placeholder symbol `__occ_0`, `__occ_1`, ... while recording syntactic context.
    """
    raw_occurrences: List[Tuple[str, str, Tuple[str, str]]] = []

    def _walk(node: sp.Expr, path: List[str]) -> sp.Expr:
        if isinstance(node, sp.Symbol):
            occ_id = f"__occ_{len(raw_occurrences)}"
            ctx = _infer_context_labels(path)
            raw_occurrences.append((occ_id, node.name, ctx))
            return sp.Symbol(occ_id, real=True)

        if not node.args:
            return node

        # Determine child path annotations for meaningful context descriptions
        if node.func == sp.exp:
            new_args = [_walk(arg, path + ["exp"]) for arg in node.args]
            return sp.exp(*new_args, evaluate=False)

        if node.func == sp.log:
            new_args = [_walk(arg, path + ["log"]) for arg in node.args]
            return sp.log(*new_args, evaluate=False)

        if node.func == sp.gamma:
            new_args = [_walk(arg, path + ["gamma"]) for arg in node.args]
            return sp.gamma(*new_args, evaluate=False)

        if node.func == sp.binomial:
            new_args = [_walk(arg, path + ["binomial"]) for arg in node.args]
            return sp.binomial(*new_args, evaluate=False)

        if isinstance(node, sp.Pow):
            base, exp_part = node.args
            is_sqrt = exp_part == sp.Rational(1, 2) or exp_part == sp.Rational(-1, 2)
            is_neg = _is_negative_power(node)
            base_path = list(path)
            if is_neg:
                base_path.append("denom")
            if is_sqrt:
                base_path.append("sqrt")
            new_base = _walk(base, base_path)
            new_exp = _walk(exp_part, path + ["power_exp"])
            return sp.Pow(new_base, new_exp, evaluate=False)

        if isinstance(node, sp.Mul):
            has_denom = any(_is_negative_power(a) for a in node.args)
            new_args = []
            for arg in node.args:
                if _is_negative_power(arg):
                    new_args.append(_walk(arg, path))  # Pow handler adds "denom"
                elif has_denom:
                    new_args.append(_walk(arg, path + ["numer"]))
                else:
                    new_args.append(_walk(arg, path + ["factor"]))
            return sp.Mul(*new_args, evaluate=False)

        new_args = [_walk(arg, path) for arg in node.args]
        return node.func(*new_args, evaluate=False)

    tagged_expr = _walk(expr, [])

    # Count total occurrences per variable
    totals: Dict[str, int] = {}
    for _, var_name, _ in raw_occurrences:
        totals[var_name] = totals.get(var_name, 0) + 1

    seen_counts: Dict[str, int] = {}
    occ_map: Dict[str, OccurrenceMeta] = {}
    ordered_ids: List[str] = []

    for occ_id, var_name, (ctx_no, ctx_en) in raw_occurrences:
        seen_counts[var_name] = seen_counts.get(var_name, 0) + 1
        occ_map[occ_id] = OccurrenceMeta(
            occ_id=occ_id,
            var_name=var_name,
            display_name=display_names.get(var_name, pretty_symbol_name(var_name)),
            occurrence_index=seen_counts[var_name],
            total_occurrences=totals[var_name],
            context_no=ctx_no,
            context_en=ctx_en,
            is_independent=(var_name == independent_var),
        )
        ordered_ids.append(occ_id)

    return tagged_expr, occ_map, ordered_ids


def _safe_lambdify(symbols: List[sp.Symbol], expr: sp.Expr) -> Optional[Callable[..., Any]]:
    """Compile a SymPy expression into a fast NumPy/SciPy callable."""
    try:
        return sp.lambdify(symbols, expr, modules=_NUMPY_SCIPY_MODULES)
    except Exception:
        return None


def _reevaluate_expr(node: sp.Expr) -> sp.Expr:
    """Rebuild an evaluate=False expression with standard SymPy evaluation for clean cancellation."""
    if not node.args:
        return node
    return node.func(*[_reevaluate_expr(a) for a in node.args])


def _compile_sensitivity_functions(
    expr_tagged: sp.Expr,
    occ_ids_ordered: List[str],
) -> Tuple[Dict[str, Optional[Callable[..., Any]]], Dict[str, Optional[Callable[..., Any]]]]:
    """
    Pre-compile exact symbolic partial derivative and elasticity functions for every
    tagged occurrence. Uses symbolic cancellation (s * diff(f, s) / f) so exponential
    underflow cancels out analytically!
    """
    occ_symbols = [sp.Symbol(oid, real=True) for oid in occ_ids_ordered]
    elasticity_fns: Dict[str, Optional[Callable[..., Any]]] = {}
    partial_fns: Dict[str, Optional[Callable[..., Any]]] = {}

    # Re-evaluate into canonical SymPy form (unique __occ_i symbols stay distinct!)
    # and rewrite binomial/factorial in terms of gamma for smooth differentiation.
    diff_base_expr = _reevaluate_expr(expr_tagged).replace(
        sp.binomial,
        lambda n, k: sp.gamma(n + 1) / (sp.gamma(k + 1) * sp.gamma(n - k + 1)),
    ).replace(sp.factorial, lambda k: sp.gamma(k + 1))

    for sym in occ_symbols:
        oid = sym.name
        try:
            d_expr = sp.diff(diff_base_expr, sym)
            if d_expr.has(sp.Derivative):
                partial_fns[oid] = None
                elasticity_fns[oid] = None
                continue
            partial_fns[oid] = _safe_lambdify(occ_symbols, d_expr)

            # Symbolic elasticity: (sym * d_expr) / diff_base_expr -> cancels exp(...) factors!
            e_expr = sp.cancel(sym * d_expr / diff_base_expr)
            elasticity_fns[oid] = _safe_lambdify(occ_symbols, e_expr)
        except Exception:
            partial_fns[oid] = None
            elasticity_fns[oid] = None

    return partial_fns, elasticity_fns


def _infer_output_type(
    lhs_raw: str,
    expr: sp.Expr,
    variables: List[str],
) -> Tuple[str, Optional[str]]:
    """
    Automatically classify what kind of output a custom formula produces:
    - 'discrete_func': e.g. p(k) = binomial(n, k) * ...
    - 'continuous_func': e.g. f(x) = ... or y = 2*x + 1
    - 'scalar': e.g. b = 1/a, tfidf(t, d) = tf(t, d) * log(N/df(t))
    """
    var_set = set(variables)
    uses_discrete_ops = bool(expr.has(sp.binomial) or expr.has(sp.factorial))

    # Check if LHS is of the form func(arg) where arg is an actual RHS variable
    m = re.match(r"^\s*[A-Za-z][A-Za-z0-9_]*\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)\s*$", lhs_raw)
    if m:
        arg_name = _normalize_greek_input(m.group(1))
        if arg_name in var_set:
            if uses_discrete_ops or arg_name in {"k", "m"}:
                return "discrete_func", arg_name
            return "continuous_func", arg_name

    # If 'x' is a standalone symbol on RHS, treat as continuous function of x
    if "x" in var_set:
        if uses_discrete_ops:
            return "discrete_func", "x"
        return "continuous_func", "x"

    # If 'k' is present alongside binomial/factorial, treat as discrete function of k
    if "k" in var_set and uses_discrete_ops:
        return "discrete_func", "k"

    return "scalar", None


def _smart_default_value(var_name: str, display_name: str, expr: sp.Expr) -> float:
    """Choose sensible positive/non-singular default values for arbitrary custom variables."""
    lower = display_name.lower()
    if lower in {"n", "n_docs", "total"} or var_name == "N":
        return 100.0
    if "df" in lower:
        return 10.0
    if "tf" in lower:
        return 3.0
    if lower in {"p", "p(a)", "p(b)", "p(b|a)", "prob"} or lower.startswith("p("):
        return 0.5
    if lower in {"mu", "μ", "x0", "z0", "c", "b"}:
        # Check if variable appears in a denominator or log
        sym = sp.Symbol(var_name, real=True)
        if _appears_as_divisor_or_log(expr, sym):
            return 2.0
        return 1.0
    if lower in {"sigma", "σ", "sd", "std"}:
        return 1.0
    if lower in {"k", "x", "t", "z"}:
        return 1.0
    if lower in {"a", "alpha", "beta", "lam", "lambda"}:
        return 2.0
    return 2.0


def _appears_as_divisor_or_log(expr: sp.Expr, sym: sp.Symbol) -> bool:
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and _is_negative_power(node) and node.base.has(sym):
            return True
        if node.func == sp.log and node.has(sym):
            return True
    return False


def build_parsed_formula(
    raw_formula: str,
    expr: sp.Expr,
    lhs_display: str,
    display_names: Dict[str, str],
    output_type: str,
    independent_var: Optional[str],
    default_values: Optional[Dict[str, float]] = None,
) -> ParsedFormula:
    """Build a ParsedFormula from a pre-constructed SymPy expression (used by built-in distributions & custom parser)."""
    # Preserve deterministic variable ordering: independent_var first, then alphabetical
    free_names = sorted([s.name for s in expr.free_symbols])
    if independent_var and independent_var in free_names:
        free_names.remove(independent_var)
        variables = [independent_var] + free_names
    else:
        variables = free_names

    full_display = {
        v: display_names.get(v, pretty_symbol_name(v)) for v in variables
    }

    expr_tagged, occ_map, occ_ids_ordered = tag_expression_occurrences(
        expr=expr,
        display_names=full_display,
        independent_var=independent_var,
    )

    var_symbols = [sp.Symbol(v, real=True) for v in variables]
    occ_symbols = [sp.Symbol(oid, real=True) for oid in occ_ids_ordered]

    eval_fn = _safe_lambdify(var_symbols, expr)
    tagged_eval_fn = _safe_lambdify(occ_symbols, expr_tagged)

    if eval_fn is None or tagged_eval_fn is None:
        raise ValueError("Could not compile numerical evaluator for expression.")

    partial_fns, elasticity_fns = _compile_sensitivity_functions(expr_tagged, occ_ids_ordered)

    defaults: Dict[str, float] = {}
    for v in variables:
        if default_values and v in default_values:
            defaults[v] = float(default_values[v])
        else:
            defaults[v] = _smart_default_value(v, full_display[v], expr)

    return ParsedFormula(
        raw_formula=raw_formula,
        lhs_display=lhs_display,
        expr=expr,
        expr_tagged=expr_tagged,
        variables=variables,
        display_names=full_display,
        occurrences=occ_map,
        output_type=output_type,
        independent_var=independent_var,
        eval_fn=eval_fn,
        tagged_eval_fn=tagged_eval_fn,
        occ_ids_ordered=occ_ids_ordered,
        elasticity_fns=elasticity_fns,
        partial_fns=partial_fns,
        default_values=defaults,
    )


def parse_custom_formula(raw_formula: str) -> ParsedFormula:
    """Parse an arbitrary user-entered mathematical formula into a ParsedFormula."""
    cleaned = raw_formula.strip()
    if not cleaned:
        raise ValueError("Formelen kan ikke være tom / Formula cannot be empty.")

    if not _ALLOWED_RE.match(cleaned.replace("−", "-").replace("×", "*").replace("·", "*")):
        raise ValueError("Formelen inneholder ugyldige tegn / Formula contains unsupported characters.")

    if "=" in cleaned:
        lhs_raw, rhs_raw = cleaned.split("=", 1)
        lhs_display = lhs_raw.strip() or "y"
    else:
        lhs_raw = ""
        rhs_raw = cleaned
        lhs_display = "y"

    if not rhs_raw.strip():
        raise ValueError("Høyresiden av formelen mangler / Right-hand side of formula is empty.")

    processed_rhs, compound_map = _preprocess_rhs(rhs_raw)
    local_dict = _build_local_dict(processed_rhs, compound_map)
    safe_global_dict: Dict[str, Any] = {
        "Symbol": sp.Symbol,
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
        "Function": sp.Function,
    }

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    try:
        expr = parse_expr(
            processed_rhs,
            local_dict=local_dict,
            global_dict=safe_global_dict,
            transformations=transformations,
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(str(exc)) from exc

    if not isinstance(expr, sp.Expr):
        raise ValueError("Uttrykket kunne ikke tolkes som en matematisk formel.")

    # Force real symbols for all free symbols so SymPy stays tidy
    subs_real = {s: sp.Symbol(s.name, real=True) for s in expr.free_symbols}
    if subs_real:
        expr = expr.subs(subs_real)

    free_names = sorted([s.name for s in expr.free_symbols])
    output_type, ind_var = _infer_output_type(lhs_raw, expr, free_names)

    if not lhs_raw:
        lhs_display = f"f({pretty_symbol_name(ind_var)})" if ind_var else "y"

    display_names: Dict[str, str] = {}
    for name in free_names:
        if name in compound_map:
            display_names[name] = compound_map[name]
        else:
            display_names[name] = pretty_symbol_name(name)

    return build_parsed_formula(
        raw_formula=cleaned,
        expr=expr,
        lhs_display=lhs_display,
        display_names=display_names,
        output_type=output_type,
        independent_var=ind_var,
    )


def _to_finite_float(val: Any) -> Optional[float]:
    try:
        arr = np.asarray(val)
        if arr.ndim > 0:
            val = arr.item()
        if isinstance(val, complex):
            if abs(val.imag) > 1e-9:
                return None
            val = val.real
        fval = float(val)
        if math.isfinite(fval):
            return fval
        return None
    except Exception:
        return None


def compute_smooth_scale(
    ratio: float,
    min_scale: float = 0.62,
    max_scale: float = 2.05,
    amplitude: float = 0.85,
    steepness: float = 0.65,
) -> float:
    """
    Smoothly map a positive magnitude ratio (where 1.0 is baseline) to a readable
    visual CSS font-size multiplier using tanh(ln(ratio)).
    """
    if not math.isfinite(ratio) or ratio <= 0:
        return min_scale
    log_r = math.log(max(ratio, 1e-6))
    raw = 1.0 + amplitude * math.tanh(steepness * log_r)
    return float(max(min_scale, min(max_scale, raw)))


def evaluate_formula_point(
    parsed: ParsedFormula,
    values: Dict[str, float],
    scaling_mode: str = "sensitivity",
    baseline_output: Optional[float] = None,
    baseline_values: Optional[Dict[str, float]] = None,
    baseline_partials: Optional[Dict[str, float]] = None,
) -> EvaluationResult:
    """
    Evaluate the formula at a single parameter vector and compute per-occurrence
    derivatives, elasticities, visual scale factors, and directions.
    """
    occ_args = [float(values[parsed.occurrences[oid].var_name]) for oid in parsed.occ_ids_ordered]

    try:
        raw_out = parsed.tagged_eval_fn(*occ_args)
    except Exception as exc:
        return EvaluationResult(
            output_value=float("nan"),
            response_scale=1.0,
            response_direction="neutral",
            occurrences={},
            valid=False,
            error_message=str(exc),
        )

    output_val = _to_finite_float(raw_out)
    if output_val is None:
        return EvaluationResult(
            output_value=float("nan"),
            response_scale=1.0,
            response_direction="neutral",
            occurrences={},
            valid=False,
            error_message="Non-finite output",
        )

    # Response visual scale relative to baseline
    ref_out = baseline_output if (baseline_output is not None and abs(baseline_output) > 1e-12) else 1.0
    if abs(output_val) < 1e-12:
        response_scale = 0.68
    else:
        response_scale = compute_smooth_scale(abs(output_val) / abs(ref_out), min_scale=0.65, max_scale=2.05)

    if output_val > ref_out * 1.01:
        response_direction = "pos"
    elif output_val < ref_out * 0.99:
        response_direction = "neg"
    else:
        response_direction = "neutral"

    # Compute partial derivatives & elasticities for every occurrence
    raw_partials: Dict[str, float] = {}
    raw_elasticities: Dict[str, float] = {}

    for idx, oid in enumerate(parsed.occ_ids_ordered):
        val_i = occ_args[idx]

        # 1. Partial derivative (symbolic first, central-difference fallback)
        p_val: Optional[float] = None
        p_fn = parsed.partial_fns.get(oid)
        if p_fn is not None:
            try:
                p_val = _to_finite_float(p_fn(*occ_args))
            except Exception:
                p_val = None

        if p_val is None:
            p_val = _numerical_partial(parsed.tagged_eval_fn, occ_args, idx)

        if p_val is None:
            p_val = 0.0
        raw_partials[oid] = p_val

        # 2. Elasticity (symbolic first so exp(...) underflow cancels cleanly!)
        e_val: Optional[float] = None
        e_fn = parsed.elasticity_fns.get(oid)
        if e_fn is not None:
            try:
                e_val = _to_finite_float(e_fn(*occ_args))
            except Exception:
                e_val = None

        # Regularize when parameter value is ~0 (e.g., mu = 0) so shifting mu still shows real sensitivity
        if abs(val_i) < 1e-6:
            char_scale = 1.0
            if abs(output_val) > 1e-15:
                e_val = (char_scale * p_val) / abs(output_val)
            else:
                e_val = char_scale * p_val
        elif e_val is None:
            if abs(output_val) > 1e-15:
                e_val = (val_i * p_val) / output_val
            else:
                e_val = val_i * p_val

        if e_val is None or not math.isfinite(e_val):
            e_val = 0.0
        raw_elasticities[oid] = e_val

    # Determine visual scales & relative shares
    non_ind_ids = [oid for oid in parsed.occ_ids_ordered if not parsed.occurrences[oid].is_independent]
    single_param_formula = len(non_ind_ids) == 1

    total_abs_eff = sum(abs(raw_elasticities[oid]) for oid in parsed.occ_ids_ordered)
    if total_abs_eff < 1e-12:
        total_abs_eff = 1.0

    analyses: Dict[str, OccurrenceAnalysis] = {}

    for idx, oid in enumerate(parsed.occ_ids_ordered):
        meta = parsed.occurrences[oid]
        val_i = occ_args[idx]
        p_val = raw_partials[oid]
        e_val = raw_elasticities[oid]

        if p_val > 1e-9:
            direction = "pos"
        elif p_val < -1e-9:
            direction = "neg"
        else:
            direction = "neutral"

        if scaling_mode == "value":
            ref_v = 1.0
            if baseline_values and meta.var_name in baseline_values and abs(baseline_values[meta.var_name]) > 1e-9:
                ref_v = abs(baseline_values[meta.var_name])
            elif meta.var_name in parsed.default_values and abs(parsed.default_values[meta.var_name]) > 1e-9:
                ref_v = abs(parsed.default_values[meta.var_name])

            if abs(val_i) < 1e-9:
                scale = 0.65
            else:
                scale = compute_smooth_scale(abs(val_i) / ref_v, min_scale=0.62, max_scale=2.05)
        else:
            # Sensitivity / Effect mode
            if single_param_formula and oid in non_ind_ids:
                # For 1-parameter formulas like b = 1/a where elasticity is constant (-1),
                # scale by local sensitivity |∂y/∂a| relative to baseline so it responds dynamically!
                ref_p = 1.0
                if baseline_partials and oid in baseline_partials and abs(baseline_partials[oid]) > 1e-12:
                    ref_p = abs(baseline_partials[oid])
                eff_ratio = abs(p_val) / ref_p if ref_p > 1e-12 else abs(e_val)
                scale = compute_smooth_scale(eff_ratio, min_scale=0.62, max_scale=2.05)
            else:
                # Multi-occurrence formulas (Normal, Gamma, Binomial, TF-IDF, etc.):
                # Scale directly by dimensionless elasticity |E_i|!
                if abs(e_val) < 1e-6:
                    scale = 0.65
                else:
                    scale = compute_smooth_scale(abs(e_val), min_scale=0.62, max_scale=2.05)

        analyses[oid] = OccurrenceAnalysis(
            meta=meta,
            value=val_i,
            partial_derivative=p_val,
            elasticity=e_val,
            direction=direction,
            scale=scale,
            relative_share=abs(e_val) / total_abs_eff,
        )

    return EvaluationResult(
        output_value=output_val,
        response_scale=response_scale,
        response_direction=response_direction,
        occurrences=analyses,
        valid=True,
    )


def _numerical_partial(
    fn: Callable[..., Any],
    args: List[float],
    idx: int,
) -> Optional[float]:
    """Central finite-difference derivative fallback for a single occurrence argument."""
    x0 = args[idx]
    h = max(1e-5, abs(x0) * 1e-5)

    args_plus = list(args)
    args_minus = list(args)
    args_plus[idx] = x0 + h
    args_minus[idx] = x0 - h

    f_plus = _safe_eval_scalar(fn, args_plus)
    f_minus = _safe_eval_scalar(fn, args_minus)

    if f_plus is not None and f_minus is not None:
        return (f_plus - f_minus) / (2.0 * h)

    # One-sided fallback near domain boundaries (e.g. x = 0)
    f_zero = _safe_eval_scalar(fn, args)
    if f_plus is not None and f_zero is not None:
        return (f_plus - f_zero) / h
    if f_minus is not None and f_zero is not None:
        return (f_zero - f_minus) / h
    return None


def _safe_eval_scalar(fn: Callable[..., Any], args: List[float]) -> Optional[float]:
    try:
        return _to_finite_float(fn(*args))
    except Exception:
        return None


def evaluate_curve(
    parsed: ParsedFormula,
    sweep_var: str,
    x_grid: np.ndarray,
    fixed_values: Dict[str, float],
) -> np.ndarray:
    """Evaluate a parsed formula along a 1D numpy grid for `sweep_var`."""
    y_out = np.full(len(x_grid), np.nan, dtype=float)
    for i, xv in enumerate(x_grid):
        arg_list = [
            float(xv) if v == sweep_var else float(fixed_values[v])
            for v in parsed.variables
        ]
        val = _safe_eval_scalar(parsed.eval_fn, arg_list)
        if val is not None:
            y_out[i] = val
    return y_out
