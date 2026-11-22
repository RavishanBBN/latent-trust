"""Correctness tests for the filter, plant Jacobian, JEPA Jacobian and chi-square bands.

Run:  pytest -q
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from scipy import stats

from lbf import quadtank as qt
from lbf.autodiff import Tensor
from lbf.ekf import EKF
from lbf.jepa import JEPAWorldModel
from lbf.metrics import chi2_band, nees


# ---------------------------------------------------------------- helpers
def fd_jacobian(f, x, eps=1e-6):
    """Central finite-difference Jacobian of f: R^n -> R^m (column-vector convention)."""
    x = np.asarray(x, float)
    cols = []
    for i in range(len(x)):
        dx = np.zeros_like(x); dx[i] = eps
        cols.append((f(x + dx) - f(x - dx)) / (2 * eps))
    return np.stack(cols, axis=1)


def linear_system(seed=0):
    rng = np.random.default_rng(seed)
    n, m, k = 3, 2, 1
    A = np.array([[0.95, 0.10, 0.0], [0.0, 0.90, 0.05], [0.02, 0.0, 0.97]])
    B = rng.normal(size=(n, k)) * 0.1
    C = rng.normal(size=(m, n))
    R = np.diag([0.01, 0.02, 0.015])          # process noise (course notation)
    Q = np.diag([0.05, 0.08])                 # measurement noise
    return A, B, C, R, Q


def kalman_filter(A, B, C, R, Q, mu0, S0, U, Z):
    """Textbook linear KF (Probabilistic Robotics, Table 3.1), standard covariance update."""
    mu, S = mu0.copy(), S0.copy()
    out_mu, out_S = [], []
    for u, z in zip(U, Z):
        mu_bar = A @ mu + B @ u
        S_bar = A @ S @ A.T + R
        K = S_bar @ C.T @ np.linalg.inv(C @ S_bar @ C.T + Q)
        mu = mu_bar + K @ (z - C @ mu_bar)
        S = (np.eye(len(mu)) - K @ C) @ S_bar
        out_mu.append(mu); out_S.append(S)
    return np.array(out_mu), np.array(out_S)


def simulate_linear(A, B, C, R, Q, T, seed):
    rng = np.random.default_rng(seed)
    n, k = B.shape
    x = rng.multivariate_normal(np.zeros(n), np.eye(n))
    X, U, Z = [], [], []
    for _ in range(T):
        u = rng.normal(size=k)
        x = A @ x + B @ u + rng.multivariate_normal(np.zeros(n), R)
        z = C @ x + rng.multivariate_normal(np.zeros(len(Q)), Q)
        X.append(x); U.append(u); Z.append(z)
    return np.array(X), np.array(U), np.array(Z)


# ---------------------------------------------------------------- EKF == KF on a linear system
@pytest.mark.parametrize("joseph", [True, False])
def test_ekf_equals_kf_on_linear_system(joseph):
    A, B, C, R, Q = linear_system()
    _, U, Z = simulate_linear(A, B, C, R, Q, T=200, seed=1)
    mu0, S0 = np.zeros(3), np.eye(3)
    ekf = EKF(g=lambda u, x: A @ x + B @ u, G=lambda u, x: A,
              h=lambda x: C @ x, H=lambda x: C, R=R, Q=Q, joseph=joseph)
    out = ekf.run(mu0, S0, U, Z)
    mu_kf, S_kf = kalman_filter(A, B, C, R, Q, mu0, S0, U, Z)
    np.testing.assert_allclose(out["mu"], mu_kf, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(out["Sigma"], S_kf, rtol=1e-8, atol=1e-10)


def test_ekf_missing_measurement_is_prediction_only():
    A, B, C, R, Q = linear_system()
    ekf = EKF(g=lambda u, x: A @ x + B @ u, G=lambda u, x: A,
              h=lambda x: C @ x, H=lambda x: C, R=R, Q=Q)
    mu, S, u = np.ones(3), np.eye(3), np.array([0.3])
    r = ekf.step(mu, S, u, None)
    np.testing.assert_allclose(r.mu, A @ mu + B @ u)
    np.testing.assert_allclose(r.Sigma, A @ S @ A.T + R)
    assert np.isnan(r.nis)


def test_linear_kf_is_consistent():
    """On a correctly specified linear model, the average NEES and NIS must match their dof."""
    A, B, C, R, Q = linear_system()
    ekf = EKF(g=lambda u, x: A @ x + B @ u, G=lambda u, x: A,
              h=lambda x: C @ x, H=lambda x: C, R=R, Q=Q)
    nees_all, nis_all = [], []
    for seed in range(20):
        X, U, Z = simulate_linear(A, B, C, R, Q, T=300, seed=100 + seed)
        out = ekf.run(np.zeros(3), np.eye(3), U, Z)
        nees_all.append(nees(out["mu"][50:] - X[50:], out["Sigma"][50:]))
        nis_all.append(out["nis"][50:])
    anees, anis = np.mean(nees_all), np.mean(nis_all)
    assert abs(anees - 3.0) < 0.15, anees
    assert abs(anis - 2.0) < 0.10, anis


# ---------------------------------------------------------------- Jacobians vs finite differences
@pytest.mark.parametrize("x", [qt.H_OP, np.array([8.0, 15.0, 3.0, 0.9]), np.array([20.0, 6.0, 0.5, 4.0])])
def test_quadtank_G_disc_matches_finite_differences(x):
    u, dt = qt.U_OP, 1.0
    G = qt.G_disc(u, x, dt)
    G_fd = fd_jacobian(lambda xx: qt.g_disc(u, xx, dt), x)
    np.testing.assert_allclose(G, G_fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_jepa_predict_jacobian_matches_finite_differences(seed):
    rng = np.random.default_rng(seed)
    model = JEPAWorldModel(in_dim=40, u_dim=2, d=12, seed=seed)
    # make the predictor non-trivial (the default output layer is scaled down by 0.1)
    for W, b in model.pred:
        W.data = W.data * 5.0 + rng.normal(size=W.data.shape) * 0.1
        b.data = rng.normal(size=b.data.shape) * 0.3
    s, u = rng.normal(size=12), rng.normal(size=2)
    G = model.predict_jacobian(s, u)
    G_fd = fd_jacobian(lambda ss: model.predict(ss, u), s)
    np.testing.assert_allclose(G, G_fd, rtol=1e-5, atol=1e-7)


def test_jepa_numpy_and_autodiff_forward_agree():
    rng = np.random.default_rng(0)
    model = JEPAWorldModel(in_dim=40, u_dim=2, d=8, seed=0)
    x, u = rng.normal(size=(5, 40)), rng.normal(size=(5, 2))
    # at initialisation the EMA target equals the online encoder
    np.testing.assert_allclose(model.encode_t(Tensor(x)).data, model.encode_target(x), atol=1e-12)
    s = model.encode_target(x)
    np.testing.assert_allclose(model.predict_t(Tensor(s), Tensor(u)).data, model.predict(s, u), atol=1e-12)


# ---------------------------------------------------------------- chi-square bands
def test_chi2_band_single_run_is_plain_quantile():
    lo, hi = chi2_band(dof_per_run=4, n_runs=1, alpha=0.05)
    np.testing.assert_allclose([lo, hi], stats.chi2.ppf([0.025, 0.975], 4))


def test_chi2_band_regression_values_from_report():
    """Bands used in the EN4594 report (50 Monte Carlo runs): NEES dof 4, NIS dof 2."""
    np.testing.assert_allclose(chi2_band(4, 50), (3.2546, 4.8212), atol=1e-4)
    np.testing.assert_allclose(chi2_band(2, 50), (1.4844, 2.5912), atol=1e-4)


def test_chi2_band_has_nominal_coverage():
    rng = np.random.default_rng(0)
    dof, n_runs, trials = 3, 50, 20000
    lo, hi = chi2_band(dof, n_runs, alpha=0.05)
    avg = rng.chisquare(dof, size=(trials, n_runs)).mean(axis=1)
    coverage = np.mean((avg >= lo) & (avg <= hi))
    assert abs(coverage - 0.95) < 0.01, coverage
