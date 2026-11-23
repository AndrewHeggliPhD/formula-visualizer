"""Built-in statistical probability distributions with symbolic PDFs/PMFs and SciPy evaluators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import scipy.stats as stats
import sympy as sp

from src.formula_engine import ParsedFormula, build_parsed_formula


@dataclass(frozen=True)
class ParamSpec:
    """Slider specification for one distribution parameter."""

    name: str
    symbol: str
    label_no: str
    label_en: str
    min_val: float
    max_val: float
    default: float
    step: float
    is_int: bool = False


@dataclass
class DistributionSpec:
    """Complete definition of a built-in statistical distribution."""

    key: str
    name_no: str
    name_en: str
    description_no: str
    description_en: str
    insight_no: str
    insight_en: str
    kind: str  # "continuous" or "discrete"
    indep_var: str  # "x" or "k"
    lhs_display: str
    params: List[ParamSpec]
    default_focus: float
    build_expr: Callable[[], sp.Expr]
    scipy_dist: Callable[[Dict[str, float]], object]
    domain_fn: Callable[[Dict[str, float]], Tuple[float, float]]
    focus_bounds_fn: Callable[[Dict[str, float]], Tuple[float, float, float]]


def _normal_expr() -> sp.Expr:
    x, mu, sigma = sp.symbols("x mu sigma", real=True)
    return (1 / (sigma * sp.sqrt(2 * sp.pi))) * sp.exp(-((x - mu) ** 2) / (2 * sigma**2))


def _gamma_expr() -> sp.Expr:
    x, alpha, beta = sp.symbols("x alpha beta", real=True)
    return (beta**alpha / sp.gamma(alpha)) * x ** (alpha - 1) * sp.exp(-beta * x)


def _binomial_expr() -> sp.Expr:
    k, n, p = sp.symbols("k n p", real=True)
    return sp.binomial(n, k) * (p**k) * ((1 - p) ** (n - k))


def _poisson_expr() -> sp.Expr:
    k, lam = sp.symbols("k lam", real=True)
    return (lam**k) * sp.exp(-lam) / sp.factorial(k)


def _exponential_expr() -> sp.Expr:
    x, lam = sp.symbols("x lam", real=True)
    return lam * sp.exp(-lam * x)


def _beta_expr() -> sp.Expr:
    x, alpha, beta = sp.symbols("x alpha beta", real=True)
    return (sp.gamma(alpha + beta) / (sp.gamma(alpha) * sp.gamma(beta))) * (
        x ** (alpha - 1)
    ) * ((1 - x) ** (beta - 1))


def _student_t_expr() -> sp.Expr:
    x, nu = sp.symbols("x nu", real=True)
    return (
        sp.gamma((nu + 1) / 2)
        / (sp.sqrt(nu * sp.pi) * sp.gamma(nu / 2))
        * (1 + (x**2) / nu) ** (-(nu + 1) / 2)
    )


def _chi2_expr() -> sp.Expr:
    x, k = sp.symbols("x k", real=True)
    return (1 / ((2 ** (k / 2)) * sp.gamma(k / 2))) * (x ** (k / 2 - 1)) * sp.exp(-x / 2)


def _lognormal_expr() -> sp.Expr:
    x, mu, sigma = sp.symbols("x mu sigma", real=True)
    return (1 / (x * sigma * sp.sqrt(2 * sp.pi))) * sp.exp(
        -((sp.log(x) - mu) ** 2) / (2 * sigma**2)
    )


def _uniform_expr() -> sp.Expr:
    x, a, b = sp.symbols("x a b", real=True)
    # Include x trivially or keep exact 1/(b - a)
    _ = x
    return 1 / (b - a)


DISTRIBUTIONS: Dict[str, DistributionSpec] = {
    "normal": DistributionSpec(
        key="normal",
        name_no="Normalfordeling",
        name_en="Normal Distribution",
        description_no="Klokkekurven – den mest sentrale kontinuerlige fordelingen i statistikk.",
        description_en="The bell curve — the most central continuous distribution in statistics.",
        insight_no=(
            "σ opptrer to steder: i normaliseringsleddet foran (senker toppen) og i eksponenten "
            "(sprer halen). Dra i σ-slideren og se dem bytte dominans ved σ = |x₀ − μ|!"
        ),
        insight_en=(
            "σ appears twice: in the leading normalization factor (lowers peak) and in the exponent "
            "(widens tails). Drag σ and watch them trade dominance right at σ = |x₀ − μ|!"
        ),
        kind="continuous",
        indep_var="x",
        lhs_display="f(x)",
        params=[
            ParamSpec("mu", "μ", "Forventningsverdi (μ)", "Mean (μ)", -5.0, 5.0, 0.0, 0.1),
            ParamSpec("sigma", "σ", "Standardavvik (σ)", "Std. Dev. (σ)", 0.25, 4.0, 1.0, 0.05),
        ],
        default_focus=1.2,
        build_expr=_normal_expr,
        scipy_dist=lambda p: stats.norm(loc=p["mu"], scale=p["sigma"]),
        domain_fn=lambda p: (min(-6.0, p["mu"] - 4 * p["sigma"]), max(6.0, p["mu"] + 4 * p["sigma"])),
        focus_bounds_fn=lambda p: (
            round(p["mu"] - 3.5 * p["sigma"], 2),
            round(p["mu"] + 3.5 * p["sigma"], 2),
            0.05,
        ),
    ),
    "gamma": DistributionSpec(
        key="gamma",
        name_no="Gammafordeling",
        name_en="Gamma Distribution",
        description_no="Modell for positive skjeve variabler, f.eks. ventetider og skadebeløp.",
        description_en="Models positive right-skewed quantities such as waiting times and claim sizes.",
        insight_no=(
            "Både α (form) og β (rate) opptrer flere steder! β^α og exp(−βx) drar i hver sin retning, "
            "og skifter balanse ved x₀ = α / β."
        ),
        insight_en=(
            "Both α (shape) and β (rate) appear multiple times! β^α and exp(−βx) pull in opposite "
            "directions, crossing balance at x₀ = α / β."
        ),
        kind="continuous",
        indep_var="x",
        lhs_display="f(x)",
        params=[
            ParamSpec("alpha", "α", "Formparameter (α)", "Shape (α)", 0.5, 10.0, 2.5, 0.1),
            ParamSpec("beta", "β", "Rateparameter (β)", "Rate (β)", 0.2, 5.0, 1.0, 0.1),
        ],
        default_focus=2.0,
        build_expr=_gamma_expr,
        scipy_dist=lambda p: stats.gamma(a=p["alpha"], scale=1.0 / p["beta"]),
        domain_fn=lambda p: (0.01, max(10.0, (p["alpha"] / p["beta"]) + 5 * math.sqrt(p["alpha"]) / p["beta"])),
        focus_bounds_fn=lambda p: (
            0.1,
            round(max(8.0, (p["alpha"] / p["beta"]) + 4 * math.sqrt(p["alpha"]) / p["beta"]), 2),
            0.1,
        ),
    ),
    "binomial": DistributionSpec(
        key="binomial",
        name_no="Binomisk fordeling",
        name_en="Binomial Distribution",
        description_no="Antall suksesser k i n uavhengige forsøk med suksessannsynlighet p.",
        description_en="Number of successes k in n independent trials with success probability p.",
        insight_no=(
            "p opptrer i både p^k (trekker opp) og (1−p)^(n−k) (trekker ned). "
            "Når k₀ > n·p dominerer p^k; når k₀ < n·p dominerer (1−p)^(n−k)!"
        ),
        insight_en=(
            "p appears in both p^k (pulls up) and (1−p)^(n−k) (pulls down). "
            "When k₀ > n·p, p^k dominates; when k₀ < n·p, (1−p)^(n−k) dominates!"
        ),
        kind="discrete",
        indep_var="k",
        lhs_display="P(X = k)",
        params=[
            ParamSpec("n", "n", "Antall forsøk (n)", "Number of trials (n)", 2, 40, 12, 1, is_int=True),
            ParamSpec("p", "p", "Suksessannsynlighet (p)", "Success probability (p)", 0.05, 0.95, 0.4, 0.02),
        ],
        default_focus=6.0,
        build_expr=_binomial_expr,
        scipy_dist=lambda p: stats.binom(n=int(round(p["n"])), p=p["p"]),
        domain_fn=lambda p: (0.0, float(int(round(p["n"])))),
        focus_bounds_fn=lambda p: (0.0, float(int(round(p["n"]))), 1.0),
    ),
    "poisson": DistributionSpec(
        key="poisson",
        name_no="Poissonfordeling",
        name_en="Poisson Distribution",
        description_no="Antall hendelser k i et fast tidsrom når gjennomsnittlig rate er λ.",
        description_en="Count of events k in a fixed interval with average rate λ.",
        insight_no=(
            "λ opptrer både i λ^k (øker sannsynligheten) og i exp(−λ) (senker den). "
            "Likevekten ligger nøyaktig ved k₀ = λ!"
        ),
        insight_en=(
            "λ appears in both λ^k (pulls probability up) and exp(−λ) (pulls it down). "
            "Exact equilibrium occurs at k₀ = λ!"
        ),
        kind="discrete",
        indep_var="k",
        lhs_display="P(X = k)",
        params=[
            ParamSpec("lam", "λ", "Rate / snitt (λ)", "Rate / mean (λ)", 0.5, 20.0, 4.5, 0.25),
        ],
        default_focus=6.0,
        build_expr=_poisson_expr,
        scipy_dist=lambda p: stats.poisson(mu=p["lam"]),
        domain_fn=lambda p: (0.0, float(max(15, int(math.ceil(p["lam"] + 4.5 * math.sqrt(p["lam"])))))),
        focus_bounds_fn=lambda p: (
            0.0,
            float(max(15, int(math.ceil(p["lam"] + 4.0 * math.sqrt(p["lam"]))))),
            1.0,
        ),
    ),
    "exponential": DistributionSpec(
        key="exponential",
        name_no="Eksponentialfordeling",
        name_en="Exponential Distribution",
        description_no="Tid mellom uavhengige hendelser i en Poisson-prosess med rate λ.",
        description_en="Time between independent events in a Poisson process with rate λ.",
        insight_no=(
            "λ står både foran (øker tettheten) og i eksponenten −λx (senker tettheten). "
            "Krysningspunktet er nøyaktig x₀ = 1/λ!"
        ),
        insight_en=(
            "λ sits both in front (increases density) and in the exponent −λx (decreases density). "
            "Their crossover is exactly x₀ = 1/λ!"
        ),
        kind="continuous",
        indep_var="x",
        lhs_display="f(x)",
        params=[
            ParamSpec("lam", "λ", "Rateparameter (λ)", "Rate (λ)", 0.2, 5.0, 1.0, 0.05),
        ],
        default_focus=1.5,
        build_expr=_exponential_expr,
        scipy_dist=lambda p: stats.expon(scale=1.0 / p["lam"]),
        domain_fn=lambda p: (0.0, max(6.0, 5.5 / p["lam"])),
        focus_bounds_fn=lambda p: (0.05, round(max(5.0, 4.5 / p["lam"]), 2), 0.05),
    ),
    "beta": DistributionSpec(
        key="beta",
        name_no="Betafordeling",
        name_en="Beta Distribution",
        description_no="Fleksibel fordeling på intervallet (0, 1), mye brukt for sannsynligheter i Bayesiansk statistikk.",
        description_en="Flexible distribution on (0, 1), widely used as a prior for probabilities in Bayesian statistics.",
        insight_no=(
            "α styrer tyngden mot 1 og β styrer tyngden mot 0, samtidig som begge påvirker normaliseringsleddet Γ(α+β)/(Γ(α)Γ(β))."
        ),
        insight_en=(
            "α pulls mass toward 1 and β pulls mass toward 0, while both also reshape the normalizing factor Γ(α+β)/(Γ(α)Γ(β))."
        ),
        kind="continuous",
        indep_var="x",
        lhs_display="f(x)",
        params=[
            ParamSpec("alpha", "α", "Form α", "Shape α", 0.4, 10.0, 2.0, 0.1),
            ParamSpec("beta", "β", "Form β", "Shape β", 0.4, 10.0, 5.0, 0.1),
        ],
        default_focus=0.35,
        build_expr=_beta_expr,
        scipy_dist=lambda p: stats.beta(a=p["alpha"], b=p["beta"]),
        domain_fn=lambda p: (0.01, 0.99),
        focus_bounds_fn=lambda p: (0.02, 0.98, 0.02),
    ),
    "student_t": DistributionSpec(
        key="student_t",
        name_no="Student's t-fordeling",
        name_en="Student's t Distribution",
        description_no="Symmetrisk klokkekurve med tyngre haler enn normalfordelingen – styrt av frihetsgrader ν.",
        description_en="Symmetric bell curve with heavier tails than Normal — controlled by degrees of freedom ν.",
        insight_no=(
            "Frihetsgradene ν opptrer både i normaliseringsfaktoren og i haleleddet (1 + x²/ν)^(−(ν+1)/2). "
            "Jo større ν er, desto mer ligner den normalfordelingen."
        ),
        insight_en=(
            "Degrees of freedom ν appear both in the normalizing factor and in the tail term (1 + x²/ν)^(−(ν+1)/2). "
            "As ν grows, it approaches the standard Normal."
        ),
        kind="continuous",
        indep_var="x",
        lhs_display="f(x)",
        params=[
            ParamSpec("nu", "ν", "Frihetsgrader (ν)", "Degrees of freedom (ν)", 1.0, 30.0, 3.0, 0.5),
        ],
        default_focus=1.8,
        build_expr=_student_t_expr,
        scipy_dist=lambda p: stats.t(df=p["nu"]),
        domain_fn=lambda p: (-6.0, 6.0),
        focus_bounds_fn=lambda p: (-5.0, 5.0, 0.1),
    ),
    "chi2": DistributionSpec(
        key="chi2",
        name_no="Kjikvadratfordeling (χ²)",
        name_en="Chi-Squared Distribution (χ²)",
        description_no="Sum av kvadrerte standard normalvariabler med k frihetsgrader – brukt i hypotesetester.",
        description_en="Sum of squared standard normal variables with k degrees of freedom — key for hypothesis testing.",
        insight_no=(
            "Frihetsgradene k påvirker både skalering (2^(k/2)·Γ(k/2)) og potensen x^(k/2 − 1), som flytter toppen mot høyre."
        ),
        insight_en=(
            "Degrees of freedom k act in both the scaling denominator 2^(k/2)·Γ(k/2) and x^(k/2 − 1), shifting the peak right."
        ),
        kind="continuous",
        indep_var="x",
        lhs_display="f(x)",
        params=[
            ParamSpec("k", "k", "Frihetsgrader (k)", "Degrees of freedom (k)", 1.0, 20.0, 4.0, 0.5),
        ],
        default_focus=3.5,
        build_expr=_chi2_expr,
        scipy_dist=lambda p: stats.chi2(df=p["k"]),
        domain_fn=lambda p: (0.05, max(12.0, p["k"] + 4.5 * math.sqrt(2.0 * p["k"]))),
        focus_bounds_fn=lambda p: (
            0.1,
            round(max(10.0, p["k"] + 4.0 * math.sqrt(2.0 * p["k"])), 2),
            0.1,
        ),
    ),
    "lognormal": DistributionSpec(
        key="lognormal",
        name_no="Lognormalfordeling",
        name_en="Log-Normal Distribution",
        description_no="Fordelingen til X = exp(Y) når Y er normalfordelt – typisk for inntekt og aksjepriser.",
        description_en="Distribution of X = exp(Y) when Y is Normal — common for incomes and asset prices.",
        insight_no=(
            "Akkurat som normalfordelingen opptrer σ både i faktoren foran og i eksponenten med (ln(x) − μ)²."
        ),
        insight_en=(
            "Just like the Normal distribution, σ appears both in the leading factor and in the exponent with (ln(x) − μ)²."
        ),
        kind="continuous",
        indep_var="x",
        lhs_display="f(x)",
        params=[
            ParamSpec("mu", "μ", "Log-snitt (μ)", "Log-mean (μ)", -1.5, 2.0, 0.0, 0.1),
            ParamSpec("sigma", "σ", "Log-standardavvik (σ)", "Log-std (σ)", 0.2, 1.8, 0.6, 0.05),
        ],
        default_focus=1.6,
        build_expr=_lognormal_expr,
        scipy_dist=lambda p: stats.lognorm(s=p["sigma"], scale=math.exp(p["mu"])),
        domain_fn=lambda p: (0.02, min(40.0, max(6.0, math.exp(p["mu"] + 2.6 * p["sigma"])))),
        focus_bounds_fn=lambda p: (
            0.05,
            round(min(30.0, max(5.0, math.exp(p["mu"] + 2.2 * p["sigma"]))), 2),
            0.05,
        ),
    ),
    "uniform": DistributionSpec(
        key="uniform",
        name_no="Uniform (rektangulær) fordeling",
        name_en="Uniform Distribution",
        description_no="Lik sannsynlighetstetthet over hele intervallet [a, b].",
        description_en="Constant probability density across the interval [a, b].",
        insight_no=(
            "Tettheten 1/(b − a) avhenger kun av bredden (b − a): øker du b synker høyden, øker du a stiger høyden!"
        ),
        insight_en=(
            "Density 1/(b − a) depends solely on width (b − a): increasing b lowers height, increasing a raises height!"
        ),
        kind="continuous",
        indep_var="x",
        lhs_display="f(x)",
        params=[
            ParamSpec("a", "a", "Nedre grense (a)", "Lower bound (a)", -5.0, 3.0, 0.0, 0.2),
            ParamSpec("b", "b", "Øvre grense (b)", "Upper bound (b)", -2.0, 8.0, 4.0, 0.2),
        ],
        default_focus=2.0,
        build_expr=_uniform_expr,
        scipy_dist=lambda p: stats.uniform(loc=min(p["a"], p["b"] - 0.1), scale=max(0.1, p["b"] - p["a"])),
        domain_fn=lambda p: (min(p["a"], p["b"]) - 1.5, max(p["a"], p["b"]) + 1.5),
        focus_bounds_fn=lambda p: (
            round(min(p["a"], p["b"]) + 0.05, 2),
            round(max(p["a"], p["b"]) - 0.05, 2),
            0.05,
        ),
    ),
}


def get_distribution_parsed(spec: DistributionSpec) -> ParsedFormula:
    """Build a ParsedFormula for a built-in distribution."""
    expr = spec.build_expr()
    display_names = {p.name: p.symbol for p in spec.params}
    display_names[spec.indep_var] = spec.indep_var
    defaults = {p.name: float(p.default) for p in spec.params}
    defaults[spec.indep_var] = float(spec.default_focus)

    output_type = "continuous_dist" if spec.kind == "continuous" else "discrete_dist"
    return build_parsed_formula(
        raw_formula=f"{spec.lhs_display} = {sp.sstr(expr)}",
        expr=expr,
        lhs_display=spec.lhs_display,
        display_names=display_names,
        output_type=output_type,
        independent_var=spec.indep_var,
        default_values=defaults,
    )


def compute_distribution_stats(spec: DistributionSpec, param_values: Dict[str, float]) -> Dict[str, float]:
    """Return mean, variance, and standard deviation using SciPy."""
    dist = spec.scipy_dist(param_values)
    try:
        mean_val = float(dist.mean())
    except Exception:
        mean_val = float("nan")
    try:
        var_val = float(dist.var())
    except Exception:
        var_val = float("nan")
    std_val = math.sqrt(var_val) if (math.isfinite(var_val) and var_val >= 0) else float("nan")
    return {"mean": mean_val, "var": var_val, "std": std_val}


def evaluate_distribution_curve(
    spec: DistributionSpec,
    param_values: Dict[str, float],
    view_type: str = "pdf",
    num_points: int = 350,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate PDF/PMF or CDF over an automatically chosen domain."""
    dist = spec.scipy_dist(param_values)
    x_min, x_max = spec.domain_fn(param_values)

    if spec.kind == "discrete":
        x_grid = np.arange(int(math.floor(x_min)), int(math.ceil(x_max)) + 1, dtype=float)
        if view_type == "cdf":
            y_grid = np.asarray(dist.cdf(x_grid), dtype=float)
        else:
            y_grid = np.asarray(dist.pmf(x_grid), dtype=float)
    else:
        x_grid = np.linspace(x_min, x_max, num_points)
        if view_type == "cdf":
            y_grid = np.asarray(dist.cdf(x_grid), dtype=float)
        else:
            y_grid = np.asarray(dist.pdf(x_grid), dtype=float)

    return x_grid, y_grid
