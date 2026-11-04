"""Extended Kalman filter in the EN4594 / Probabilistic Robotics notation.

    x_t = g(u_t, x_{t-1}) + eps_t,   eps_t ~ N(0, R_t)   (process noise covariance R)
    z_t = h(x_t)          + delta_t, delta_t ~ N(0, Q_t) (measurement noise covariance Q)

Algorithm Extended_Kalman_filter(mu_{t-1}, Sigma_{t-1}, u_t, z_t):
  1: mu_bar    = g(u_t, mu_{t-1})
  2: Sigma_bar = G_t Sigma_{t-1} G_t^T + R_t
  3: K_t       = Sigma_bar H_t^T (H_t Sigma_bar H_t^T + Q_t)^-1
  4: mu_t      = mu_bar + K_t (z_t - h(mu_bar))
  5: Sigma_t   = (I - K_t H_t) Sigma_bar            (Joseph form used for stability)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class StepResult:
    mu: np.ndarray
    Sigma: np.ndarray
    mu_bar: np.ndarray
    Sigma_bar: np.ndarray
    innovation: np.ndarray | None   # z_t - h(mu_bar)
    S: np.ndarray | None            # innovation covariance H Sigma_bar H^T + Q
    nis: float                      # normalised innovation squared (nan if no measurement)


class EKF:
    def __init__(self, g: Callable, G: Callable, h: Callable, H: Callable,
                 R: np.ndarray, Q: np.ndarray, joseph: bool = True):
        self.g, self.G, self.h, self.H = g, G, h, H
        self.R, self.Q, self.joseph = R, Q, joseph

    def step(self, mu, Sigma, u, z) -> StepResult:
        # ---- prediction (lines 1-2)
        Gt = self.G(u, mu)
        mu_bar = self.g(u, mu)
        Sigma_bar = Gt @ Sigma @ Gt.T + self.R
        Sigma_bar = 0.5 * (Sigma_bar + Sigma_bar.T)          # keep symmetric
        if z is None or np.any(np.isnan(z)):                  # no measurement: prediction only
            return StepResult(mu_bar, Sigma_bar, mu_bar, Sigma_bar, None, None, np.nan)
        # ---- correction (lines 3-5)
        Ht = self.H(mu_bar)
        nu = z - self.h(mu_bar)                               # innovation
        S = Ht @ Sigma_bar @ Ht.T + self.Q
        K = np.linalg.solve(S.T, (Sigma_bar @ Ht.T).T).T      # Sigma_bar H^T S^-1
        mu_t = mu_bar + K @ nu
        I_KH = np.eye(len(mu)) - K @ Ht
        if self.joseph:
            Sigma_t = I_KH @ Sigma_bar @ I_KH.T + K @ self.Q @ K.T
        else:
            Sigma_t = I_KH @ Sigma_bar
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)
        nis = float(nu @ np.linalg.solve(S, nu))
        return StepResult(mu_t, Sigma_t, mu_bar, Sigma_bar, nu, S, nis)

    def run(self, mu0, Sigma0, U, Z):
        T, n = len(U), len(mu0)
        out = dict(mu=np.empty((T, n)), Sigma=np.empty((T, n, n)), nis=np.full(T, np.nan),
                   innov=np.full((T, Z.shape[1]), np.nan), S=np.full((T, Z.shape[1], Z.shape[1]), np.nan),
                   mu_bar=np.empty((T, n)))
        mu, Sigma = np.asarray(mu0, float), np.asarray(Sigma0, float)
        for t in range(T):
            r = self.step(mu, Sigma, U[t], Z[t])
            mu, Sigma = r.mu, r.Sigma
            out["mu"][t], out["Sigma"][t], out["nis"][t], out["mu_bar"][t] = mu, Sigma, r.nis, r.mu_bar
            if r.innovation is not None:
                out["innov"][t], out["S"][t] = r.innovation, r.S
        return out
