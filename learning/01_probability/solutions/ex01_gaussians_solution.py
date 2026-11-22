"""Lesson 1 exercises: SOLUTION. Try the exercise file first!

Run from the repo root:  .venv/bin/python learning/01_probability/solutions/ex01_gaussians_solution.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# Two tank levels (cm) with correlated uncertainty
MU = np.array([12.4, 12.7])
SIGMA = np.array([[1.0, 0.6],
                  [0.6, 0.5]])
X = rng.multivariate_normal(MU, SIGMA, size=20000)      # shape (N, 2): one sample per row


# =============================================================== Exercise 1: covariance
def sample_mean_and_cov(X):
    """Return (mean vector of shape (2,), covariance matrix of shape (2, 2)) estimated from samples X (N, 2)."""
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)          # rowvar=False: columns are variables, rows are samples
    return mean, cov


def correlation(Sigma):
    """Return the correlation coefficient rho between variable 0 and variable 1."""
    return Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])


# =============================================================== Exercise 2: linear transform (predict step)
A = np.array([[0.9, 0.1],
              [0.0, 0.95]])
B_OFFSET = np.array([0.5, 0.2])
R = np.eye(2) * 0.05                       # process noise covariance (course notation)


def predict_gaussian(mu, Sigma, A, b, R):
    """Mean and covariance of y = A x + b + eps, with x ~ N(mu, Sigma) and eps ~ N(0, R)."""
    return A @ mu + b, A @ Sigma @ A.T + R


def transform_samples(X, A, b, R, rng):
    """Apply y = A x + b + eps to every row of X (eps drawn fresh per row). Return Y with shape (N, 2)."""
    eps = rng.multivariate_normal(np.zeros(2), R, size=len(X))
    return X @ A.T + b + eps               # row form of A x: x^T A^T


# =============================================================== Exercise 3: Bayes update, grid vs formula
MU0, VAR0 = 12.0, 1.0 ** 2                 # prior: level ~ N(12, 1)
Z, VAR_Z = 12.6, 0.5 ** 2                  # sensor reading and its noise variance


def bayes_on_grid(mu0, var0, z, var_z):
    """Brute-force Bayes: multiply prior and likelihood on a grid, normalise.
    Return (grid, posterior_pdf, posterior_mean, posterior_var)."""
    x = np.linspace(6, 18, 20001)
    dx = x[1] - x[0]
    prior = stats.norm.pdf(x, mu0, np.sqrt(var0))
    likelihood = stats.norm.pdf(z, x, np.sqrt(var_z))      # p(z | x) as a function of x
    post = prior * likelihood
    post /= post.sum() * dx                                  # normalise so it integrates to 1
    mean = (x * post).sum() * dx
    var = ((x - mean) ** 2 * post).sum() * dx
    return x, post, mean, var


def kalman_update_1d(mu0, var0, z, var_z):
    """Return (K, posterior_mean, posterior_var) using the 1-D Kalman formulas."""
    K = var0 / (var0 + var_z)
    return K, mu0 + K * (z - mu0), (1 - K) * var0


# =============================================================== Exercise 4: chi-squared and NIS
def mahalanobis_sq(X, mu, Sigma):
    """Return m_i = (x_i - mu)^T Sigma^{-1} (x_i - mu) for every row x_i of X. Shape (N,)."""
    E = X - mu
    return np.einsum("ni,ni->n", E, np.linalg.solve(Sigma, E.T).T)


def alarm_fraction(m, dof, alpha=0.01):
    """Fraction of values in m above the (1 - alpha) chi-squared quantile with `dof` degrees of freedom."""
    return float(np.mean(m > stats.chi2.ppf(1 - alpha, dof)))


# =============================================================== checks (do not edit below)
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main():
    # ---- Exercise 1
    mean, cov = sample_mean_and_cov(X)
    check("1a sample mean", np.allclose(mean, MU, atol=0.03), f"{np.round(mean, 3)}")
    check("1a sample cov", np.allclose(cov, SIGMA, atol=0.03), f"\n{np.round(cov, 3)}")
    rho = correlation(SIGMA)
    check("1b correlation", np.isclose(rho, 0.6 / np.sqrt(0.5)), f"rho = {rho:.3f}")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:2000, 0], X[:2000, 1], s=2, alpha=0.4)
    vals, vecs = np.linalg.eigh(SIGMA)
    th = np.linspace(0, 2 * np.pi, 200)
    for k in (1, 2):
        ell = MU[:, None] + vecs @ (k * np.sqrt(vals)[:, None] * np.vstack([np.cos(th), np.sin(th)]))
        ax.plot(*ell, "k", lw=1)
    ax.set(xlabel="h1 (cm)", ylabel="h2 (cm)", title="1- and 2-sigma ellipses")
    fig.savefig(FIG / "ex1_ellipse.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ---- Exercise 2
    mu_y, S_y = predict_gaussian(MU, SIGMA, A, B_OFFSET, R)
    Y = transform_samples(X, A, B_OFFSET, R, np.random.default_rng(1))
    check("2a predicted mean matches samples", np.allclose(mu_y, Y.mean(0), atol=0.03), f"{np.round(mu_y, 3)}")
    check("2b predicted cov matches samples", np.allclose(S_y, np.cov(Y, rowvar=False), atol=0.03),
          f"\n{np.round(S_y, 3)}")
    check("2c prediction adds uncertainty (trace grows when R is big enough)",
          np.trace(S_y) > np.trace(A @ SIGMA @ A.T))

    # ---- Exercise 3
    x, post, m_grid, v_grid = bayes_on_grid(MU0, VAR0, Z, VAR_Z)
    K, m_kf, v_kf = kalman_update_1d(MU0, VAR0, Z, VAR_Z)
    check("3a grid posterior mean = Kalman mean", np.isclose(m_grid, m_kf, atol=1e-4), f"{m_grid:.4f} vs {m_kf:.4f}")
    check("3b grid posterior var  = Kalman var", np.isclose(v_grid, v_kf, atol=1e-4), f"{v_grid:.4f} vs {v_kf:.4f}")
    check("3c gain", np.isclose(K, 0.8), f"K = {K:.3f}")
    check("3d posterior is tighter than both prior and sensor", v_kf < min(VAR0, VAR_Z))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, stats.norm.pdf(x, MU0, np.sqrt(VAR0)), label="prior")
    ax.plot(x, stats.norm.pdf(x, Z, np.sqrt(VAR_Z)), label="likelihood (as fn of x)")
    ax.plot(x, post, "k", lw=2, label="posterior")
    ax.set(xlim=(9, 15), xlabel="tank level (cm)"); ax.legend(fontsize=8)
    fig.savefig(FIG / "ex3_bayes.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ---- Exercise 4
    m = mahalanobis_sq(X, MU, SIGMA)
    check("4a mean of Mahalanobis^2 is about d = 2", abs(m.mean() - 2) < 0.05, f"mean = {m.mean():.3f}")
    fa = alarm_fraction(m, 2)
    check("4b ~1% exceed the 99% chi2 threshold (correct model)", abs(fa - 0.01) < 0.003, f"{100 * fa:.2f}%")
    m_bad = mahalanobis_sq(X, MU, SIGMA / 4)        # overconfident: claims 2x smaller std than reality
    fa_bad = alarm_fraction(m_bad, 2)
    check("4c overconfident covariance -> mean ~ 8 and many false alarms", abs(m_bad.mean() - 8) < 0.2,
          f"mean = {m_bad.mean():.2f}, alarms = {100 * fa_bad:.1f}%")
    fig, ax = plt.subplots(figsize=(5, 3))
    grid = np.linspace(0, 20, 400)
    ax.hist(m, bins=100, range=(0, 20), density=True, alpha=0.5, label="correct Sigma")
    ax.hist(m_bad, bins=100, range=(0, 20), density=True, alpha=0.5, label="Sigma/4 (overconfident)")
    ax.plot(grid, stats.chi2.pdf(grid, 2), "k", label=r"$\chi^2_2$ pdf")
    ax.axvline(stats.chi2.ppf(0.99, 2), color="r", ls="--", label="99% threshold")
    ax.set(xlabel="Mahalanobis$^2$ (this is what NIS is)"); ax.legend(fontsize=8)
    fig.savefig(FIG / "ex4_chi2.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigures saved to {FIG}")


if __name__ == "__main__":
    main()


# =============================================================== answers to the self-check questions
"""
1. Prediction is Sigma_bar = G Sigma G^T + R. The "+ R" (process noise) term adds uncertainty every step, because the
   model is never perfect. (G Sigma G^T can shrink or grow the ellipse; R always inflates it.)

2. K = s0^2 / (s0^2 + sz^2): a noisier sensor means a bigger sz^2, so K goes DOWN (we trust the sensor less).
   The posterior variance (1-K) s0^2 goes UP: we learned less from the measurement.

3. The off-diagonal covariance terms between h3 and the measured h1 (tank 3 drains into tank 1). The Kalman gain
   K = Sigma_bar H^T S^-1 has a non-zero row for h3 exactly because Sigma_bar[h3, h1] != 0.

4. With 2 measurements, the average NIS should be about 2. A value of 6 means the innovations are ~sqrt(3) times larger than
   the filter claims, so the filter is OVERCONFIDENT: S = H Sigma_bar H^T + Q is too small. Q (sensor noise) or R
   (process noise) is set too low, or the model itself is wrong (model mismatch acts like extra unmodelled noise).

5. The JEPA predictor's error (tr R_lat = 0.072) is larger than the encoder's noise (tr Q_lat = 0.023), so the
   Kalman gain is close to 1 and the filter mostly just copies the raw encoder measurement.
"""
