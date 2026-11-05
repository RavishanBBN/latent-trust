"""Filter evaluation: RMSE, NEES/NIS consistency (chi-square bounds), NIS-based change detection."""
from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(est, truth, axis=0):
    return np.sqrt(np.nanmean((est - truth) ** 2, axis=axis))


def nees(err, Sigma):
    """Normalised estimation error squared e^T Sigma^-1 e for each time step. err (T,n), Sigma (T,n,n)."""
    return np.einsum("ti,ti->t", err, np.linalg.solve(Sigma, err[..., None])[..., 0])


def chi2_band(dof_per_run, n_runs, alpha=0.05):
    """Two-sided (1-alpha) band for the Monte-Carlo average of a chi2(dof) statistic over n_runs runs.

    n_runs * avg ~ chi2(n_runs * dof)  =>  band = chi2.ppf([a/2, 1-a/2], n_runs*dof) / n_runs.
    """
    lo, hi = stats.chi2.ppf([alpha / 2, 1 - alpha / 2], n_runs * dof_per_run) / n_runs
    return float(lo), float(hi)


def windowed_nis_test(nis, window, dof, alpha=0.01):
    """Sliding-window sum of NIS ~ chi2(window*dof) under 'model correct'. Returns statistic, threshold, alarms."""
    x = np.nan_to_num(nis, nan=float(dof))          # missing steps contribute their expectation
    s = np.convolve(x, np.ones(window), mode="full")[: len(x)]
    thr = stats.chi2.ppf(1 - alpha, window * dof)
    alarm = s > thr
    alarm[: window - 1] = False                      # wait until the window is full
    return s, thr, alarm


def cusum_nis(nis, dof, k=None, h=None):
    """One-sided CUSUM on NIS (Page 1954; Basseville & Nikiforov 1993). Detects a mean increase."""
    x = np.nan_to_num(nis, nan=float(dof))
    k = dof + 0.5 * np.sqrt(2 * dof) if k is None else k   # drift ~ half a std above the mean
    g = np.zeros_like(x)
    for t in range(1, len(x)):
        g[t] = max(0.0, g[t - 1] + x[t] - k)
    return g


def detection_delay(alarm, fault_start):
    after = np.flatnonzero(alarm[fault_start:])
    return int(after[0]) if after.size else None


def false_alarm_rate(alarm, fault_start):
    pre = alarm[:fault_start]
    return float(pre.mean()) if pre.size else np.nan
