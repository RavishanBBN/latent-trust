"""A tiny reverse-mode automatic-differentiation engine on NumPy arrays.

Only what the JEPA world model needs: matmul, broadcasting add/sub/mul, tanh, relu, power,
sqrt, sum/mean, transpose and concatenation. Written because PyTorch is not available in the
assignment environment; the research codebase should switch to PyTorch/JAX.
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` over broadcast dimensions so that it matches `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, parents=(), backward=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._parents = parents
        self._backward = backward
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    # ---------------- graph construction helpers
    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def _acc(self, g):
        self.grad = g if self.grad is None else self.grad + g

    # ---------------- ops
    def __add__(self, o):
        o = Tensor._wrap(o)
        out = Tensor(self.data + o.data, (self, o))
        def bw():
            self._acc(_unbroadcast(out.grad, self.data.shape)); o._acc(_unbroadcast(out.grad, o.data.shape))
        out._backward = bw
        return out
    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-Tensor._wrap(o))

    def __rsub__(self, o):
        return Tensor._wrap(o) - self

    def __mul__(self, o):
        o = Tensor._wrap(o)
        out = Tensor(self.data * o.data, (self, o))
        def bw():
            self._acc(_unbroadcast(out.grad * o.data, self.data.shape))
            o._acc(_unbroadcast(out.grad * self.data, o.data.shape))
        out._backward = bw
        return out
    __rmul__ = __mul__

    def __matmul__(self, o):
        o = Tensor._wrap(o)
        out = Tensor(self.data @ o.data, (self, o))
        def bw():
            self._acc(out.grad @ o.data.T); o._acc(self.data.T @ out.grad)
        out._backward = bw
        return out

    def __pow__(self, p: float):
        out = Tensor(self.data ** p, (self,))
        def bw():
            self._acc(out.grad * p * self.data ** (p - 1))
        out._backward = bw
        return out

    def tanh(self):
        y = np.tanh(self.data)
        out = Tensor(y, (self,))
        def bw():
            self._acc(out.grad * (1.0 - y ** 2))
        out._backward = bw
        return out

    def relu(self):
        m = self.data > 0
        out = Tensor(self.data * m, (self,))
        def bw():
            self._acc(out.grad * m)
        out._backward = bw
        return out

    def sqrt(self):
        y = np.sqrt(self.data)
        out = Tensor(y, (self,))
        def bw():
            self._acc(out.grad * 0.5 / y)
        out._backward = bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.data.shape).copy())
        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    @property
    def T(self):
        out = Tensor(self.data.T, (self,))
        def bw():
            self._acc(out.grad.T)
        out._backward = bw
        return out

    # ---------------- backprop
    def backward(self):
        order, seen = [], set()
        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)
        visit(self)
        for t in order:
            t.grad = None
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            if t._backward is not None and t.grad is not None:
                t._backward()


def concat(ts, axis=1):
    ts = [Tensor._wrap(t) for t in ts]
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis), tuple(ts))
    sizes = np.cumsum([t.data.shape[axis] for t in ts])[:-1]
    def bw():
        for t, g in zip(ts, np.split(out.grad, sizes, axis=axis)):
            t._acc(g)
    out._backward = bw
    return out


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.params, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)
