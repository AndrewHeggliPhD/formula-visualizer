"""Curated library of ready-made custom formulas for exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FormulaExample:
    key: str
    title_no: str
    title_en: str
    formula: str
    description_no: str
    description_en: str
    # Optional custom defaults/ranges keyed by display name (e.g. "tf(t, d)", "N", "a")
    defaults: Dict[str, float]
    ranges: Dict[str, Tuple[float, float, float]]  # (min, max, step)


EXAMPLE_FORMULAS: List[FormulaExample] = [
    FormulaExample(
        key="tfidf",
        title_no="TF-IDF (Termfrekvens – Invers dokumentfrekvens)",
        title_en="TF-IDF (Term Frequency – Inverse Document Frequency)",
        formula="tfidf(t, d) = tf(t, d) * log(N / df(t))",
        description_no="Måler hvor viktig et ord t er i et dokument d gitt N dokumenter totalt.",
        description_en="Measures how important a term t is in document d across a corpus of N documents.",
        defaults={"tf(t, d)": 4.0, "N": 100.0, "df(t)": 10.0},
        ranges={
            "tf(t, d)": (0.5, 25.0, 0.5),
            "N": (10.0, 1000.0, 10.0),
            "df(t)": (1.0, 95.0, 1.0),
        },
    ),
    FormulaExample(
        key="inverse",
        title_no="Invers sammenheng (b = 1/a)",
        title_en="Inverse relationship (b = 1/a)",
        formula="b = 1 / a",
        description_no="Enkel skalarformel: se hvordan b og sensitiviteten til a eksploderer når a nærmer seg 0.",
        description_en="Simple scalar formula: watch b and sensitivity to a grow dramatically as a approaches 0.",
        defaults={"a": 1.0},
        ranges={"a": (0.1, 10.0, 0.1)},
    ),
    FormulaExample(
        key="bayes",
        title_no="Bayes' teorem",
        title_en="Bayes' Theorem",
        formula="P(A|B) = P(B|A) * P(A) / P(B)",
        description_no="Oppdaterer sannsynligheten for hendelse A gitt ny evidens B.",
        description_en="Updates the probability of hypothesis A given observed evidence B.",
        defaults={"P(B|A)": 0.85, "P(A)": 0.20, "P(B)": 0.35},
        ranges={
            "P(B|A)": (0.05, 1.0, 0.01),
            "P(A)": (0.01, 1.0, 0.01),
            "P(B)": (0.05, 1.0, 0.01),
        },
    ),
    FormulaExample(
        key="sigmoid",
        title_no="Sigmoid / Logistisk funksjon",
        title_en="Sigmoid / Logistic Function",
        formula="f(x) = 1 / (1 + exp(-k * (x - x0)))",
        description_no="S-kurven brukt i logistisk regresjon og nevrale nettverk.",
        description_en="The S-curve used in logistic regression and neural networks.",
        defaults={"x": 1.0, "k": 1.5, "x0": 0.0},
        ranges={
            "x": (-6.0, 6.0, 0.1),
            "k": (0.2, 5.0, 0.1),
            "x0": (-4.0, 4.0, 0.1),
        },
    ),
    FormulaExample(
        key="zscore",
        title_no="Z-skår (standardisering)",
        title_en="Z-score (Standardization)",
        formula="z = (x - mu) / sigma",
        description_no="Antall standardavvik en observasjon x ligger fra gjennomsnittet μ.",
        description_en="Number of standard deviations an observation x lies from the mean μ.",
        defaults={"x": 115.0, "mu": 100.0, "sigma": 15.0},
        ranges={
            "x": (50.0, 160.0, 1.0),
            "mu": (60.0, 140.0, 1.0),
            "sigma": (2.0, 40.0, 0.5),
        },
    ),
    FormulaExample(
        key="compound",
        title_no="Rentes rente",
        title_en="Compound Interest",
        formula="A = P * (1 + r)^t",
        description_no="Eksponentiell vekst av startkapital P ved rente r over t år.",
        description_en="Exponential growth of principal P at interest rate r over t years.",
        defaults={"P": 100.0, "r": 0.07, "t": 10.0},
        ranges={
            "P": (10.0, 500.0, 10.0),
            "r": (0.01, 0.25, 0.005),
            "t": (1.0, 40.0, 1.0),
        },
    ),
    FormulaExample(
        key="quadratic",
        title_no="Andregradsfunksjon (Parabel)",
        title_en="Quadratic Function (Parabola)",
        formula="f(x) = a * x^2 + b * x + c",
        description_no="Klassisk polynom – se hvordan a, b og c påvirker høyden i ulike x-verdier.",
        description_en="Classic polynomial — see how a, b, and c affect the curve height at different x.",
        defaults={"x": 2.0, "a": 1.0, "b": -1.0, "c": 2.0},
        ranges={
            "x": (-5.0, 5.0, 0.1),
            "a": (-3.0, 3.0, 0.1),
            "b": (-5.0, 5.0, 0.1),
            "c": (-5.0, 5.0, 0.2),
        },
    ),
    FormulaExample(
        key="normal_custom",
        title_no="Normalfordeling (som egen formel)",
        title_en="Normal Distribution (as custom formula)",
        formula="f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-((x - mu)^2) / (2 * sigma^2))",
        description_no="Demonstrerer at appens formelmotor skiller de to σ-forekomstene automatisk også i fritekst!",
        description_en="Demonstrates that the custom parser separates both σ occurrences automatically!",
        defaults={"x": 1.2, "mu": 0.0, "sigma": 1.0},
        ranges={
            "x": (-4.0, 4.0, 0.05),
            "mu": (-3.0, 3.0, 0.1),
            "sigma": (0.25, 3.5, 0.05),
        },
    ),
]
