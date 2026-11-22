"""Quadruple-tank process (Johansson, IEEE TCST 8(3):456-465, 2000).

Continuous-time model, minimum-phase setting P-:

    dh1/dt = -a1/A1 sqrt(2 g h1) + a3/A1 sqrt(2 g h3) + g1 k1/A1 v1
    dh2/dt = -a2/A2 sqrt(2 g h2) + a4/A2 sqrt(2 g h4) + g2 k2/A2 v2
    dh3/dt = -a3/A3 sqrt(2 g h3) + (1-g2) k2/A3 v2
    dh4/dt = -a4/A4 sqrt(2 g h4) + (1-g1) k1/A4 v1

State x = [h1, h2, h3, h4] (cm), control u = [v1, v2] (V), measurement z = [h1, h2] (cm).
Parameter values are those reported for P- in Johansson (2000) (confirmed via secondary sources, see docs).
"""
from __future__ import annotations

import numpy as np

PARAMS = dict(
    A=np.array([28.0, 32.0, 28.0, 32.0]),      # tank cross-sections (cm^2)
    a=np.array([0.071, 0.057, 0.071, 0.057]),  # outlet cross-sections (cm^2)
    k=np.array([3.33, 3.35]),                  # pump gains (cm^3 / (V s))
    gamma=np.array([0.70, 0.60]),              # valve split ratios (P-, minimum phase)
    g=981.0,                                   # gravity (cm/s^2)
)
H_OP = np.array([12.4, 12.7, 1.8, 1.4])      # operating point at v1 = v2 = 3.0 V
U_OP = np.array([3.0, 3.0])
H_MIN = 1e-3                                  # tanks cannot go negative


def f_cont(h, v, p=PARAMS, leak=None):
    """Continuous-time dynamics dh/dt. `leak` = (tank_index, extra_outlet_area_cm2) adds a leak."""
    A, a, k, gm, g = p["A"], p["a"], p["k"], p["gamma"], p["g"]
    hp = np.maximum(h, 0.0)
    q = a * np.sqrt(2.0 * g * hp)               # outflow of each tank (cm^3/s)
    dh = np.empty(4)
    dh[0] = (-q[0] + q[2] + gm[0] * k[0] * v[0]) / A[0]
    dh[1] = (-q[1] + q[3] + gm[1] * k[1] * v[1]) / A[1]
    dh[2] = (-q[2] + (1.0 - gm[1]) * k[1] * v[1]) / A[2]
    dh[3] = (-q[3] + (1.0 - gm[0]) * k[0] * v[0]) / A[3]
    if leak is not None:
        i, area = leak
        dh[i] -= area * np.sqrt(2.0 * g * hp[i]) / A[i]
    return dh


def jac_cont(h, p=PARAMS, eps=0.05):
    """Analytic Jacobian d f / d h of the continuous dynamics (4x4).

    d/dh_i [a_i sqrt(2 g h_i)] = a_i g / sqrt(2 g h_i). h is floored at `eps` to avoid the
    infinite slope of sqrt at h = 0.
    """
    A, a, g = p["A"], p["a"], p["g"]
    hs = np.maximum(h, eps)
    dq = a * g / np.sqrt(2.0 * g * hs)          # d q_i / d h_i
    J = np.zeros((4, 4))
    J[0, 0] = -dq[0] / A[0]; J[0, 2] = dq[2] / A[0]
    J[1, 1] = -dq[1] / A[1]; J[1, 3] = dq[3] / A[1]
    J[2, 2] = -dq[2] / A[2]
    J[3, 3] = -dq[3] / A[3]
    return J


# ---------- discrete-time model used by the EKF (forward Euler, step dt) ----------

def g_disc(u, x, dt, p=PARAMS):
    """Motion model g(u_t, x_{t-1}) = x_{t-1} + dt f(x_{t-1}, u_t)  (Euler)."""
    return np.maximum(x + dt * f_cont(x, u, p), H_MIN)


def G_disc(u, x, dt, p=PARAMS):
    """Jacobian G_t = dg/dx evaluated at mu_{t-1}:  I + dt * df/dh."""
    return np.eye(4) + dt * jac_cont(x, p)


H_MEAS = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0]])        # z = [h1, h2] (linear measurement model)


def h_meas(x):
    return H_MEAS @ x


# ---------- simulator (the "real plant": RK4 with sub-steps + process noise) ----------

def rk4_step(h, v, dt, p=PARAMS, leak=None, substeps=10):
    hh = dt / substeps
    for _ in range(substeps):
        k1 = f_cont(h, v, p, leak)
        k2 = f_cont(h + 0.5 * hh * k1, v, p, leak)
        k3 = f_cont(h + 0.5 * hh * k2, v, p, leak)
        k4 = f_cont(h + hh * k3, v, p, leak)
        h = np.maximum(h + hh / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), H_MIN)
    return h


def random_inputs(T, rng, lo=2.0, hi=4.0, hold=(30, 120)):
    """Piecewise-constant random pump voltages (a common excitation for identification)."""
    u = np.empty((T, 2))
    t = 0
    while t < T:
        L = int(rng.integers(hold[0], hold[1] + 1))
        u[t:t + L] = rng.uniform(lo, hi, size=2)
        t += L
    return u


def simulate(T=1800, dt=1.0, seed=0, sigma_w=0.01, sigma_v=0.15, u=None, h0=None,
             leak=None, leak_start=None, bias=None, bias_start=None, dropout=None):
    """Simulate one episode.

    sigma_w : process-noise std per step on each tank level (cm)  -> true R
    sigma_v : level-sensor noise std (cm)                           -> true Q
    leak    : (tank, area) applied from step `leak_start`
    bias    : (sensor, offset_cm) added to the measurement from step `bias_start`
    dropout : (t0, t1) measurements missing for t0 <= t < t1
    Returns dict with x (T,4) true states, z (T,2) measurements (NaN when missing),
    z_clean (T,2), u (T,2).
    """
    rng = np.random.default_rng(seed)
    if u is None:
        u = random_inputs(T, rng)
    h = (H_OP * rng.uniform(0.8, 1.2, 4)) if h0 is None else np.array(h0, float)
    X = np.empty((T, 4)); Z = np.empty((T, 2)); Zc = np.empty((T, 2))
    for t in range(T):
        lk = leak if (leak is not None and leak_start is not None and t >= leak_start) else None
        h = rk4_step(h, u[t], dt, leak=lk)
        h = np.maximum(h + sigma_w * rng.standard_normal(4), H_MIN)
        X[t] = h
        Zc[t] = h[:2]
        Z[t] = h[:2] + sigma_v * rng.standard_normal(2)
        if bias is not None and bias_start is not None and t >= bias_start:
            Z[t, bias[0]] += bias[1]
        if dropout is not None and dropout[0] <= t < dropout[1]:
            Z[t] = np.nan
    return dict(x=X, z=Z, z_clean=Zc, u=u, dt=dt, sigma_w=sigma_w, sigma_v=sigma_v)
