"""Paired tests and multiple-comparison corrections (numpy + scipy only)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as _st


def mcnemar(b: int, c: int) -> dict:
    """McNemar on discordant counts b (ctrl wrong -> treat right, W->C) and
    c (ctrl right -> treat wrong, C->W).

    Primary p is the EXACT two-sided binomial test (b ~ Bin(b+c, 0.5)); at our
    n (24 tasks, a handful of discordant pairs) the chi-square approximation is
    unreliable. The continuity-corrected chi-square p is reported alongside.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_exact": 1.0, "chi2_cc": 0.0, "p_chi2_cc": 1.0}
    p_exact = float(_st.binomtest(b, n, 0.5, alternative="two-sided").pvalue)
    chi2 = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "p_exact": min(1.0, p_exact), "chi2_cc": float(chi2),
            "p_chi2_cc": float(_st.chi2.sf(chi2, 1))}


def paired_bootstrap(treat, ctrl, B: int = 10000, seed: int = 20261013, ci: float = 0.95) -> dict:
    """Paired percentile bootstrap over task pairs of d = treat - ctrl (0/1 outcomes).

    Every comparison uses a fresh ``np.random.default_rng(seed)`` so results do
    not depend on comparison order. p_boot = min(1, 2*min(P(d*<=0), P(d*>=0))).
    Returns mean-difference CI and the matching net-flip CI (n * mean).
    """
    t = np.asarray(treat, dtype=float)
    c = np.asarray(ctrl, dtype=float)
    d = t - c
    n = len(d)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "mean_diff": nan, "ci_low": nan, "ci_high": nan, "net": 0,
                "net_ci_low": nan, "net_ci_high": nan, "p_boot": 1.0, "B": B, "seed": seed}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    p = min(1.0, 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    if np.all(d == 0):
        p = 1.0
    return {"n": n, "mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "net": int(d.sum()), "net_ci_low": float(lo * n), "net_ci_high": float(hi * n),
            "p_boot": float(p), "B": B, "seed": seed}


def bh_adjust(pvals) -> list:
    """Benjamini-Hochberg step-up adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return [float(min(1.0, x)) for x in q]


def holm_adjust(pvals) -> list:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return [float(x) for x in adj]


def mean_sd(xs) -> tuple:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return (float("nan"), float("nan"), 0)
    m = sum(xs) / len(xs)
    sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")
    return (m, sd, len(xs))
