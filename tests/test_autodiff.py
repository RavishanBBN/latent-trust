"""Finite-difference gradient check for the autodiff engine (run: python tests/test_autodiff.py)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from lbf.autodiff import Tensor, concat


def f(W1, b1, W2, x, u):
    z = (x @ W1 + b1).tanh()
    c = concat([z, u], axis=1)
    y = c @ W2
    zc = y - y.mean(axis=0, keepdims=True)
    var = (zc ** 2).mean(axis=0)
    std = (var + 1e-4).sqrt()
    return (y ** 2).mean() + (1.0 - std).relu().mean() + ((zc.T @ zc) ** 2).sum() * 1e-3


def test_gradcheck():
    rng = np.random.default_rng(0)
    x, u = Tensor(rng.normal(size=(7, 5))), Tensor(rng.normal(size=(7, 2)))
    P = [Tensor(rng.normal(size=s) * 0.5, requires_grad=True) for s in [(5, 4), (1, 4), (6, 3)]]
    loss = f(*P, x, u)
    loss.backward()
    for p in P:
        num = np.zeros_like(p.data)
        for idx in np.ndindex(p.data.shape):
            old = p.data[idx]
            p.data[idx] = old + 1e-6; lp = f(*P, x, u).data
            p.data[idx] = old - 1e-6; lm = f(*P, x, u).data
            p.data[idx] = old
            num[idx] = (lp - lm) / 2e-6
        err = np.abs(num - p.grad).max() / (np.abs(num).max() + 1e-12)
        assert err < 1e-5, err
    print("gradcheck OK")


if __name__ == "__main__":
    test_gradcheck()
