"""EKF in the latent space of a decoder-free JEPA world model.

Latent state-space model (Probabilistic Robotics notation):
    s_t = p_phi(s_{t-1}, u_t) + eps_t,     eps_t   ~ N(0, R_lat)   predictor = motion model g
    y_t = s_t + delta_t,                   delta_t ~ N(0, Q_lat)   y_t = f_bar(window_t) = "latent measurement"

JEPA has no decoder, so there is no observation model back to sensor space. Instead the target encoder
maps each new sensor window into the latent space, and that embedding is treated as a direct, noisy
measurement of the latent state (h(s) = s, H = I).
"""
from __future__ import annotations

import numpy as np

from .ekf import EKF


def build_latent_ekf(model, R_lat, Q_lat):
    d = model.d
    I = np.eye(d)
    return EKF(g=lambda u, s: model.predict(s, u),
               G=lambda u, s: model.predict_jacobian(s, u),
               h=lambda s: s, H=lambda s: I, R=R_lat, Q=Q_lat)


def estimate_latent_noise(model, Xw_noisy, Xw_clean, Un, idx):
    """Calibrate Q_lat and R_lat from a simulation run where noise-free sensor windows exist.

    Q_lat = Cov[f_bar(noisy window) - f_bar(clean window)]
    R_lat = Cov[f_bar(clean_{t+1}) - p(f_bar(clean_t), u_{t+1})]
    (Only possible in simulation; with real data use innovation-based adaptive estimation instead.)
    """
    yn = model.encode_target(Xw_noisy[idx])
    yc = model.encode_target(Xw_clean[idx])
    Q = np.cov((yn - yc).T)
    yc1 = model.encode_target(Xw_clean[idx + 1])
    R = np.cov((yc1 - model.predict(yc, Un[idx + 1])).T)
    return R, Q


def run_latent_ekf(lekf, Xw, Un, t_start, T):
    """Filter one episode. Xw: (T, W*m) normalised windows (NaN rows before t_start); Un: (T, 2)."""
    mu = lekf_meas(lekf, Xw, t_start)
    Sigma = lekf.Q.copy()
    d = len(mu)
    out = dict(mu=np.full((T, d), np.nan), Sigma=np.full((T, d, d), np.nan), nis=np.full(T, np.nan),
               y=np.full((T, d), np.nan))
    out["mu"][t_start], out["Sigma"][t_start] = mu, Sigma
    for t in range(t_start + 1, T):
        y = None if np.any(np.isnan(Xw[t])) else lekf.model_encode(Xw[t])
        r = lekf.step(mu, Sigma, Un[t], y)
        mu, Sigma = r.mu, r.Sigma
        out["mu"][t], out["Sigma"][t], out["nis"][t] = mu, Sigma, r.nis
        if y is not None:
            out["y"][t] = y
    return out


def lekf_meas(lekf, Xw, t):
    return lekf.model_encode(Xw[t])


def attach_encoder(lekf, model):
    lekf.model_encode = lambda x: model.encode_target(x)[0]
    return lekf
