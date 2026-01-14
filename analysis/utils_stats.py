#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils_stats.py

Utilities for statistical analysis:
- assumptions checks (normality, homoscedasticity)
- effect sizes (Cohen's d, Hedges' g, eta^2, omega^2)
- multiple testing correction
- safe wrappers returning a consistent dict payload
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Sequence, Union

import numpy as np
from scipy import stats

ArrayLike = Union[Sequence[float], np.ndarray]
Alternative = Literal["two-sided", "less", "greater"]



# -----------------------------
# Basic helpers
# -----------------------------

def _as_1d(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    a = a[~np.isnan(a)]
    return a


def describe(x: ArrayLike) -> dict[str, float]:
    """Robust-ish descriptive statistics on a 1D vector (NaNs ignored)."""
    a = _as_1d(x)
    if a.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan}
    q1, q3 = np.percentile(a, [25, 75])
    return {
        "n": float(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size >= 2 else float(np.std(a)),
        "median": float(np.median(a)),
        "iqr": float(q3 - q1),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def confidence_interval_mean(x: ArrayLike, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Student-t confidence interval for the mean.
    Returns (mean, ci_low, ci_high).
    """
    a = _as_1d(x)
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    m = float(np.mean(a))
    if a.size < 2:
        return (m, np.nan, np.nan)
    se = stats.sem(a, nan_policy="omit")
    h = se * stats.t.ppf((1 + confidence) / 2, df=a.size - 1)
    return (m, float(m - h), float(m + h))


# -----------------------------
# Assumption checks
# -----------------------------

def shapiro_test(x: ArrayLike) -> dict[str, Any]:
    """
    Shapiro-Wilk normality test.
    Note: Shapiro is not recommended for very large n; still useful here.
    """
    a = _as_1d(x)
    if a.size < 3:
        return {"test": "shapiro", "n": int(a.size), "W": np.nan, "p": np.nan, "ok": False}
    W, p = stats.shapiro(a)
    return {"test": "shapiro", "n": int(a.size), "W": float(W), "p": float(p), "ok": bool(p >= 0.05)}


def levene_test(*groups: ArrayLike, center: Literal["mean", "median", "trimmed"] = "median") -> dict[str, Any]:
    """Levene test for equal variances across groups (NaNs ignored)."""
    g = [_as_1d(x) for x in groups]
    if any(gi.size < 2 for gi in g) or len(g) < 2:
        return {"test": "levene", "k": len(g), "stat": np.nan, "p": np.nan, "ok": False}
    stat, p = stats.levene(*g, center=center)
    return {"test": "levene", "k": len(g), "stat": float(stat), "p": float(p), "ok": bool(p >= 0.05)}


# -----------------------------
# Effect sizes
# -----------------------------

def cohens_d(x: ArrayLike, y: ArrayLike) -> float:
    """Cohen's d for two independent samples."""
    a, b = _as_1d(x), _as_1d(y)
    if a.size < 2 or b.size < 2:
        return float("nan")
    nx, ny = a.size, b.size
    vx = np.var(a, ddof=1)
    vy = np.var(b, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / sp)


def hedges_g(x: ArrayLike, y: ArrayLike) -> float:
    """Hedges' g (bias-corrected Cohen's d)."""
    a, b = _as_1d(x), _as_1d(y)
    d = cohens_d(a, b)
    if np.isnan(d):
        return float("nan")
    nx, ny = a.size, b.size
    df = nx + ny - 2
    if df <= 0:
        return float("nan")
    J = 1 - (3 / (4 * df - 1))
    return float(J * d)


def eta_squared_anova(f_stat: float, df_between: float, df_within: float) -> float:
    """Eta^2 from one-way ANOVA F and dfs."""
    denom = f_stat * df_between + df_within
    if denom <= 0:
        return float("nan")
    return float((f_stat * df_between) / denom)


def omega_squared_anova(f_stat: float, df_between: float, df_within: float) -> float:
    """Omega^2 from one-way ANOVA F and dfs."""
    denom = f_stat * df_between + df_within + 1
    if denom <= 0:
        return float("nan")
    return float((df_between * (f_stat - 1)) / denom)


# -----------------------------
# Multiple testing correction
# -----------------------------

def p_adjust(p_values: Iterable[float], method: Literal["bonferroni", "holm", "bh"] = "bh") -> np.ndarray:
    """
    Adjust p-values for multiple tests.
    - bonferroni: p*m
    - holm: step-down Bonferroni
    - bh: Benjamini-Hochberg FDR
    """
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    if m == 0:
        return p

    if method == "bonferroni":
        return np.clip(p * m, 0, 1)

    if method == "holm":
        order = np.argsort(p)
        adj = np.empty_like(p)
        for i, idx in enumerate(order):
            adj[idx] = (m - i) * p[idx]
        # enforce monotonicity
        adj_sorted = adj[order]
        adj_sorted = np.maximum.accumulate(adj_sorted)
        adj[order] = np.clip(adj_sorted, 0, 1)
        return adj

    if method == "bh":
        order = np.argsort(p)
        ranked = np.empty_like(p)
        ranked[order] = np.arange(1, m + 1)
        adj = p * m / ranked
        # enforce monotonicity from largest to smallest
        adj_ordered = adj[order]
        adj_ordered[::-1] = np.minimum.accumulate(adj_ordered[::-1])
        adj[order] = np.clip(adj_ordered, 0, 1)
        return adj

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Test wrappers (consistent outputs)
# -----------------------------

def ttest_independent(
    x: ArrayLike,
    y: ArrayLike,
    equal_var: bool = False,
    alternative: Alternative = "two-sided",
) -> dict[str, Any]:
    """
    Independent samples t-test.
    By default equal_var=False (Welch), usually safer.
    """
    a, b = _as_1d(x), _as_1d(y)
    if a.size < 2 or b.size < 2:
        return {"test": "ttest_ind", "n_x": int(a.size), "n_y": int(b.size), "t": np.nan, "p": np.nan}
    res = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
    return {
        "test": "ttest_ind",
        "n_x": int(a.size),
        "n_y": int(b.size),
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "cohens_d": cohens_d(a, b),
        "hedges_g": hedges_g(a, b),
    }


def mannwhitney_u(
    x: ArrayLike,
    y: ArrayLike,
    alternative: Alternative = "two-sided",
) -> dict[str, Any]:
    """Mann-Whitney U test (non-parametric alternative for 2 independent groups)."""
    a, b = _as_1d(x), _as_1d(y)
    if a.size == 0 or b.size == 0:
        return {"test": "mannwhitneyu", "n_x": int(a.size), "n_y": int(b.size), "U": np.nan, "p": np.nan}
    res = stats.mannwhitneyu(a, b, alternative=alternative)
    return {"test": "mannwhitneyu", "n_x": int(a.size), "n_y": int(b.size), "U": float(res.statistic), "p": float(res.pvalue)}


def anova_oneway(*groups: ArrayLike) -> dict[str, Any]:
    """One-way ANOVA (assumes independent groups)."""
    g = [_as_1d(x) for x in groups]
    if len(g) < 2 or any(gi.size < 2 for gi in g):
        return {"test": "anova_oneway", "k": len(g), "F": np.nan, "p": np.nan, "eta2": np.nan, "omega2": np.nan}
    F, p = stats.f_oneway(*g)
    # Degrees of freedom:
    k = len(g)
    n_total = sum(gi.size for gi in g)
    df_between = k - 1
    df_within = n_total - k
    return {
        "test": "anova_oneway",
        "k": k,
        "F": float(F),
        "p": float(p),
        "df_between": int(df_between),
        "df_within": int(df_within),
        "eta2": eta_squared_anova(float(F), float(df_between), float(df_within)),
        "omega2": omega_squared_anova(float(F), float(df_between), float(df_within)),
    }
