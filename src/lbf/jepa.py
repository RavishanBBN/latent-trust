"""Action-conditioned JEPA world model on sensor windows (NumPy + tiny autodiff).

Encoder      f_theta : window of the last W samples of [z, u] (normalised) -> latent s in R^d
Target enc.  f_bar   : exponential moving average (EMA) of f_theta, no gradients (as in BYOL / I-JEPA)
Predictor    p_phi   : s_{t+1} = s_t + MLP([s_t, u_{t+1}])           (action-conditioned, residual)

Loss (no decoder, nothing is reconstructed in sensor space):
  L = 1/K sum_k || p^k(f_theta(x_t), u_{t+1..t+k}) - sg(f_bar(x_{t+k})) ||^2      multi-step latent prediction
      + lam_var * mean_j relu(1 - std_j(s))  + lam_cov * sum_{i!=j} Cov(s)_ij^2 / d   (VICReg anti-collapse)
"""
from __future__ import annotations

import numpy as np

from .autodiff import Adam, Tensor, concat


# ------------------------------------------------------------------ data helpers
def make_features(Z, U):
    return np.concatenate([Z, U], axis=1)          # (T, 4): [z1, z2, u1, u2]


def windows(F, W):
    """Stack windows: row t holds F[t-W+1 .. t] flattened (rows t < W-1 are invalid)."""
    T, m = F.shape
    out = np.full((T, W * m), np.nan)
    for t in range(W - 1, T):
        out[t] = F[t - W + 1:t + 1].ravel()
    return out


class Normaliser:
    def fit(self, X):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


# ------------------------------------------------------------------ model
def _init(rng, n_in, n_out, scale=1.0):
    W = rng.normal(size=(n_in, n_out)) * scale * np.sqrt(1.0 / n_in)
    return Tensor(W, requires_grad=True), Tensor(np.zeros((1, n_out)), requires_grad=True)


class JEPAWorldModel:
    def __init__(self, in_dim, u_dim=2, d=8, enc_hidden=64, pred_hidden=128, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.u_dim = d, u_dim
        self.enc = [_init(rng, in_dim, enc_hidden), _init(rng, enc_hidden, enc_hidden), _init(rng, enc_hidden, d)]
        self.pred = [_init(rng, d + u_dim, pred_hidden), _init(rng, pred_hidden, d, scale=0.1)]
        self.target = [(W.data.copy(), b.data.copy()) for W, b in self.enc]

    # ---- autodiff forward passes (training)
    def encode_t(self, x):
        h = x
        for i, (W, b) in enumerate(self.enc):
            h = h @ W + b
            if i < len(self.enc) - 1:
                h = h.tanh()
        return h

    def predict_t(self, s, u):
        (W1, b1), (W2, b2) = self.pred
        hid = (concat([s, u], axis=1) @ W1 + b1).tanh()
        return s + hid @ W2 + b2

    def params(self):
        return [p for layer in self.enc + self.pred for p in layer]

    # ---- NumPy forward passes (inference / filtering)
    def encode_target(self, x):
        h = np.atleast_2d(x)
        for i, (W, b) in enumerate(self.target):
            h = h @ W + b
            if i < len(self.target) - 1:
                h = np.tanh(h)
        return h

    def predict(self, s, u):
        (W1, b1), (W2, b2) = [(W.data, b.data) for W, b in self.pred]
        s2 = np.atleast_2d(s); u2 = np.atleast_2d(u)
        hid = np.tanh(np.concatenate([s2, u2], axis=1) @ W1 + b1)
        out = s2 + hid @ W2 + b2
        return out[0] if np.ndim(s) == 1 else out

    def predict_jacobian(self, s, u):
        """G = d p(s,u) / d s = I + W2^T diag(1 - hid^2) W1_s^T   (column-vector convention)."""
        (W1, b1), (W2, _) = [(W.data, b.data) for W, b in self.pred]
        c = np.concatenate([s, u])[None, :]
        hid = np.tanh(c @ W1 + b1)[0]
        W1s = W1[: self.d, :]                                 # (d, H)
        return np.eye(self.d) + (W2.T * (1.0 - hid ** 2)) @ W1s.T

    def ema_update(self, tau):
        for k, (W, b) in enumerate(self.enc):
            tW, tb = self.target[k]
            self.target[k] = (tau * tW + (1 - tau) * W.data, tau * tb + (1 - tau) * b.data)


# ------------------------------------------------------------------ training
def vicreg_terms(s, eps=1e-4):
    sc = s - s.mean(axis=0, keepdims=True)
    std = ((sc ** 2).mean(axis=0) + eps).sqrt()
    var_loss = (1.0 - std).relu().mean()
    n, d = s.data.shape
    cov = (sc.T @ sc) * (1.0 / (n - 1))
    off = np.ones((d, d)) - np.eye(d)
    cov_loss = ((cov * off) ** 2).sum() * (1.0 / d)
    return var_loss, cov_loss


def train_jepa(Xw, Un, valid_idx, K=5, d=8, steps=4000, batch=256, lr=1e-3, lam_var=1.0,
               lam_cov=0.04, tau=(0.99, 0.999), seed=0, log_every=250, model=None):
    """Xw: (N, W*m) normalised windows; Un: (N, 2) normalised inputs; valid_idx: indices t such that
    t .. t+K are inside the same episode (so multi-step targets exist)."""
    rng = np.random.default_rng(seed)
    model = model or JEPAWorldModel(Xw.shape[1], Un.shape[1], d=d, seed=seed)
    opt = Adam(model.params(), lr=lr)
    hist = []
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        opt.lr = lr * 0.5 * (1 + np.cos(np.pi * frac))                     # cosine decay
        idx = rng.choice(valid_idx, size=batch, replace=False)
        s0 = model.encode_t(Tensor(Xw[idx]))
        s = s0
        pred_loss = 0.0
        for k in range(1, K + 1):
            s = model.predict_t(s, Tensor(Un[idx + k]))
            tgt = model.encode_target(Xw[idx + k])                         # stop-gradient target
            pred_loss = pred_loss + ((s - tgt) ** 2).mean()
        pred_loss = pred_loss * (1.0 / K)
        v, c = vicreg_terms(s0)
        loss = pred_loss + lam_var * v + lam_cov * c
        loss.backward()
        opt.step()
        model.ema_update(tau[0] + (tau[1] - tau[0]) * frac)
        if it % log_every == 0 or it == steps - 1:
            hist.append((it, float(loss.data), float(pred_loss.data), float(v.data), float(c.data)))
    return model, np.array(hist)
