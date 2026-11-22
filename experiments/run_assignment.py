"""EN4594 mini project: EKF for the quadruple-tank process, from a physics model to a learned
latent (JEPA) world model. Produces every figure and number used in the report.

Run:  python experiments/run_assignment.py          (about 3-5 minutes on a laptop CPU)
Outputs: results/figures/*.pdf, results/results.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbf import quadtank as qt
from lbf.autodiff import Adam, Tensor
from lbf.ekf import EKF
from lbf.jepa import Normaliser, make_features, train_jepa, windows
from lbf.latent_ekf import attach_encoder, build_latent_ekf, estimate_latent_noise, run_latent_ekf
from lbf.metrics import chi2_band, nees, rmse

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = {}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                     "savefig.bbox": "tight", "lines.linewidth": 1.1})
C = dict(truth="k", meas="0.65", ekf="#1f5aa6", lat="#c2410c", raw="#7c3aed", ol="#16a34a")

DT, T = 1.0, 1800
SIG_W, SIG_V = 0.01, 0.15             # true process / measurement noise std (cm)
R_PHYS = np.eye(4) * SIG_W ** 2       # process noise covariance R (course notation)
Q_PHYS = np.eye(2) * SIG_V ** 2       # measurement noise covariance Q
MU0, SIGMA0 = qt.H_OP.copy(), np.eye(4) * 4.0
BURN = 200                             # ignore the initial transient in error statistics
N_MC = 50
TEST_SEEDS = list(range(1000, 1000 + N_MC))


def physics_ekf(R=R_PHYS, Q=Q_PHYS):
    return EKF(g=lambda u, x: qt.g_disc(u, x, DT), G=lambda u, x: qt.G_disc(u, x, DT),
               h=qt.h_meas, H=lambda x: qt.H_MEAS, R=R, Q=Q)


def open_loop(U, x0):
    X = np.empty((len(U), 4)); x = x0.copy()
    for t in range(len(U)):
        x = qt.g_disc(U[t], x, DT); X[t] = x
    return X


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png"); plt.close(fig)


# =====================================================================================
# PART A: physics-based EKF
# =====================================================================================
def part_a():
    print("Part A: physics EKF")
    ekf = physics_ekf()
    # ---- A1: one run
    d = qt.simulate(T=T, seed=TEST_SEEDS[0])
    o = ekf.run(MU0, SIGMA0, d["u"], d["z"])
    t = np.arange(T)
    fig, ax = plt.subplots(5, 1, figsize=(7, 8.2), sharex=True)
    for i in range(4):
        sd = np.sqrt(o["Sigma"][:, i, i])
        if i < 2:
            ax[i].plot(t, d["z"][:, i], ".", ms=1.2, color=C["meas"], label="measurement $z$")
        ax[i].plot(t, d["x"][:, i], color=C["truth"], label="true")
        ax[i].plot(t, o["mu"][:, i], color=C["ekf"], label=r"EKF $\mu_t$")
        ax[i].fill_between(t, o["mu"][:, i] - 2 * sd, o["mu"][:, i] + 2 * sd, color=C["ekf"], alpha=0.2, label=r"$\pm2\sigma$")
        ax[i].set_ylabel(f"$h_{i+1}$ (cm)" + ("" if i < 2 else "\n(unmeasured)"))
    ax[0].legend(ncol=4, fontsize=7, loc="upper right")
    ax[4].step(t, d["u"][:, 0], label="$v_1$"); ax[4].step(t, d["u"][:, 1], label="$v_2$")
    ax[4].set_ylabel("pump (V)"); ax[4].legend(fontsize=7); ax[4].set_xlabel("time (s)")
    save(fig, "a1_tracking")
    # zoom on h3 estimation error
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.3))
    for j, i in enumerate([0, 2]):
        e = o["mu"][:, i] - d["x"][:, i]; sd = np.sqrt(o["Sigma"][:, i, i])
        ax[j].plot(t, e, color=C["ekf"], label="error"); ax[j].plot(t, 2 * sd, "k--", lw=0.8, label=r"$\pm2\sigma$")
        ax[j].plot(t, -2 * sd, "k--", lw=0.8); ax[j].set_ylim(-0.6, 0.6); ax[j].set_title(f"$h_{i+1}$ error (cm)")
        ax[j].set_xlabel("time (s)")
    ax[0].legend(fontsize=7)
    save(fig, "a1_errors")

    # ---- A2: Monte Carlo RMSE + consistency
    errs = {"meas": [], "ol": [], "ekf": []}
    NE, NI = [], []
    for s in TEST_SEEDS:
        d = qt.simulate(T=T, seed=s)
        o = ekf.run(MU0, SIGMA0, d["u"], d["z"])
        errs["ekf"].append(rmse(o["mu"][BURN:], d["x"][BURN:]))
        errs["ol"].append(rmse(open_loop(d["u"], MU0)[BURN:], d["x"][BURN:]))
        m = np.full(4, np.nan); m[:2] = rmse(d["z"][BURN:], d["x"][BURN:, :2]); errs["meas"].append(m)
        NE.append(nees(o["mu"] - d["x"], o["Sigma"])); NI.append(o["nis"])
    tab = {k: [None if np.isnan(x) else round(float(x), 4) for x in np.nanmean(np.array(v, dtype=float), 0)] if k != 'meas' else [round(float(x), 4) for x in np.mean(np.array(v)[:, :2], 0)] + [None, None] for k, v in errs.items()}
    NE, NI = np.array(NE), np.array(NI)
    anees, anis = NE.mean(0), NI.mean(0)
    bn, bi = chi2_band(4, N_MC), chi2_band(2, N_MC)
    in_n = float(np.mean((anees[BURN:] > bn[0]) & (anees[BURN:] < bn[1])))
    in_i = float(np.mean((anis[BURN:] > bi[0]) & (anis[BURN:] < bi[1])))
    RES["A2"] = dict(rmse=tab, anees_mean=float(anees[BURN:].mean()), anis_mean=float(anis[BURN:].mean()),
                     nees_band=bn, nis_band=bi, frac_in_band_nees=in_n, frac_in_band_nis=in_i)
    print(" A2", RES["A2"])
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
    ax[0].plot(anees, color=C["ekf"]); ax[0].axhline(bn[0], ls="--", c="k", lw=.8); ax[0].axhline(bn[1], ls="--", c="k", lw=.8)
    ax[0].set_ylim(0, 10); ax[0].set_title(f"Average NEES ({N_MC} runs), n = 4"); ax[0].set_xlabel("time (s)")
    ax[1].plot(anis, color=C["ekf"]); ax[1].axhline(bi[0], ls="--", c="k", lw=.8); ax[1].axhline(bi[1], ls="--", c="k", lw=.8)
    ax[1].set_ylim(0, 5); ax[1].set_title("Average NIS, k = 2"); ax[1].set_xlabel("time (s)")
    save(fig, "a2_consistency")

    # ---- A3: sensitivity to the assumed R and Q
    r_scales = [0.25, 0.5, 1, 2, 4, 8]
    q_scales = [0.25, 1, 4]
    sens = {}
    for qs in q_scales:
        rows = []
        for rs in r_scales:
            e3, an = [], []
            for s in TEST_SEEDS[:20]:
                d = qt.simulate(T=T, seed=s)
                o = physics_ekf(R=R_PHYS * rs ** 2, Q=Q_PHYS * qs).run(MU0, SIGMA0, d["u"], d["z"])
                e3.append(rmse(o["mu"][BURN:], d["x"][BURN:]))
                an.append(nees(o["mu"] - d["x"], o["Sigma"])[BURN:].mean())
            rows.append(dict(r_scale=rs, rmse=np.mean(e3, 0).round(4).tolist(), anees=float(np.mean(an))))
        sens[str(qs)] = rows
    RES["A3"] = sens
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
    for qs, mk in zip(q_scales, "o^s"):
        rows = sens[str(qs)]
        ax[0].plot(r_scales, [np.mean(r["rmse"]) for r in rows], marker=mk, label=fr"$Q\times{qs}$")
        ax[1].plot(r_scales, [r["anees"] for r in rows], marker=mk, label=fr"$Q\times{qs}$")
    for a in ax:
        a.set_xscale("log", base=2); a.set_xlabel(r"assumed $\sigma_R$ / true $\sigma_R$")
    ax[1].set_yscale("log"); ax[1].axhline(4, c="k", ls="--", lw=.8)
    ax[0].set_ylabel("mean RMSE (cm)"); ax[1].set_ylabel("ANEES (ideal = 4)"); ax[0].legend(fontsize=7)
    save(fig, "a3_sensitivity")

    # ---- A4: measurement dropout (sensor outage 600-900 s)
    d = qt.simulate(T=T, seed=TEST_SEEDS[1], dropout=(600, 900))
    o = ekf.run(MU0, SIGMA0, d["u"], d["z"])
    tr = np.trace(o["Sigma"], axis1=1, axis2=2)
    RES["A4"] = dict(trace_before=float(tr[590]), trace_peak=float(tr[899]), trace_after=float(tr[1000]),
                     rmse_outage=rmse(o["mu"][600:900], d["x"][600:900]).round(4).tolist())
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.3))
    e = o["mu"][:, 2] - d["x"][:, 2]; sd = np.sqrt(o["Sigma"][:, 2, 2])
    ax[0].axvspan(600, 900, color="0.9"); ax[0].plot(e, color=C["ekf"]); ax[0].plot(2 * sd, "k--", lw=.8); ax[0].plot(-2 * sd, "k--", lw=.8)
    ax[0].set_title(r"$h_3$ error and $\pm2\sigma$ (outage shaded)"); ax[0].set_xlabel("time (s)"); ax[0].set_xlim(400, 1200); ax[0].set_ylim(-0.4, 0.4)
    ax[1].axvspan(600, 900, color="0.9"); ax[1].semilogy(tr, color=C["ekf"]); ax[1].set_title(r"trace($\Sigma_t$)")
    ax[1].set_xlabel("time (s)"); ax[1].set_xlim(400, 1200)
    save(fig, "a4_dropout")


# =====================================================================================
# Fault-detection helpers (shared by parts A and B)
# =====================================================================================
FAULT_T = 900
FAULTS = {"leak_tank3": dict(leak=(2, 0.02), leak_start=FAULT_T),
          "bias_sensor1": dict(bias=(0, 0.5), bias_start=FAULT_T)}
WIN = 30


def win_stat(nis, dof):
    x = np.nan_to_num(nis, nan=float(dof))
    return np.convolve(x, np.ones(WIN), mode="full")[: len(x)]


def detect(stat, thr, start):
    a = stat > thr
    a[: WIN + BURN] = False
    fa = float(a[:start].mean())
    hit = np.flatnonzero(a[start:])
    return fa, (int(hit[0]) if hit.size else None)


# =====================================================================================
# PART B: learned latent world model (JEPA) + latent EKF
# =====================================================================================
W_WIN, D_LAT, K_ROLL = 20, 12, 5


def featurise(eps, nf, clean=False):
    return [windows(nf(make_features(e["z_clean"] if clean else e["z"], e["u"])), W_WIN) for e in eps]


def part_b(ekf_phys_thr):
    print("Part B: JEPA world model + latent EKF")
    train = [qt.simulate(T=T, seed=s) for s in range(100, 130)]
    val = [qt.simulate(T=T, seed=s) for s in range(200, 205)]
    nf = Normaliser().fit(np.concatenate([make_features(e["z"], e["u"]) for e in train]))
    nu = Normaliser().fit(np.concatenate([e["u"] for e in train]))
    Xtr = np.concatenate(featurise(train, nf)); Utr = np.concatenate([nu(e["u"]) for e in train])
    valid = np.concatenate([np.arange(W_WIN - 1, T - K_ROLL) + i * T for i in range(len(train))])
    t0 = time.time()
    model, hist = train_jepa(Xtr, Utr, valid, K=K_ROLL, d=D_LAT, steps=5000, lam_cov=0.04, seed=0)
    RES["B_train_seconds"] = round(time.time() - t0, 1)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.semilogy(hist[:, 0], hist[:, 2], label="latent prediction"); ax.semilogy(hist[:, 0], hist[:, 4] * 0.04 + 1e-6, label="covariance term")
    ax.set_xlabel("iteration"); ax.legend(fontsize=7); ax.set_title("JEPA training loss")
    save(fig, "b1_training")

    # ---- readout (evaluation only): small MLP from latent to the 4 true levels
    Htr = np.concatenate([e["x"] for e in train])[valid]
    Ztr = model.encode_target(Xtr[valid])
    ym, ys = Htr.mean(0), Htr.std(0)
    rng = np.random.default_rng(1)
    P1 = Tensor(rng.normal(size=(D_LAT, 64)) / np.sqrt(D_LAT), requires_grad=True); pb1 = Tensor(np.zeros((1, 64)), requires_grad=True)
    P2 = Tensor(rng.normal(size=(64, 4)) / 8, requires_grad=True); pb2 = Tensor(np.zeros((1, 4)), requires_grad=True)
    opt = Adam([P1, pb1, P2, pb2], lr=3e-3)
    for _ in range(3000):
        i = rng.choice(len(Ztr), 512)
        out = (Tensor(Ztr[i]) @ P1 + pb1).tanh() @ P2 + pb2
        loss = ((out - (Htr[i] - ym) / ys) ** 2).mean(); loss.backward(); opt.step()
    readout = lambda Z: (np.tanh(np.atleast_2d(Z) @ P1.data + pb1.data) @ P2.data + pb2.data) * ys + ym

    # ---- calibrate latent noise covariances (simulation oracle) + one NIS-matching scale
    Xv = np.concatenate(featurise(val, nf)); Xvc = np.concatenate(featurise(val, nf, clean=True))
    Uv = np.concatenate([nu(e["u"]) for e in val])
    vv = np.concatenate([np.arange(W_WIN - 1, T - 2) + i * T for i in range(len(val))])
    R_lat, Q_lat = estimate_latent_noise(model, Xv, Xvc, Uv, vv)
    lekf = attach_encoder(build_latent_ekf(model, R_lat, Q_lat), model)
    scale = 1.0
    for _ in range(3):                         # match mean NIS to its expectation d on validation runs
        nis = []
        for e, Xe in zip(val, featurise(val, nf)):
            o = run_latent_ekf(lekf, Xe, nu(e["u"]), W_WIN - 1, T)
            nis.append(np.nanmean(o["nis"][BURN:]))
        c = np.mean(nis) / D_LAT
        scale *= c
        lekf.R, lekf.Q = lekf.R * c, lekf.Q * c
    RES["B_calibration"] = dict(nis_scale=float(scale))

    # ---- fault-free test: estimation accuracy and prediction
    est = {"raw": [], "lat": [], "phys": []}
    pred = {"raw": {}, "lat": {}, "phys": {}}
    horizons = [1, 10, 30, 60, 120]
    for k in horizons:
        for key in pred:
            pred[key][k] = []
    lat_nis = []
    ekf_p = physics_ekf()
    for s in TEST_SEEDS[:30]:
        e = qt.simulate(T=T, seed=s)
        Xe = featurise([e], nf)[0]; Ue = nu(e["u"])
        o = run_latent_ekf(lekf, Xe, Ue, W_WIN - 1, T)
        lat_nis.append(o["nis"])
        Y = model.encode_target(Xe[W_WIN - 1:])
        raw_h = np.full((T, 4), np.nan); raw_h[W_WIN - 1:] = readout(Y)
        lat_h = np.full((T, 4), np.nan); lat_h[W_WIN - 1:] = readout(o["mu"][W_WIN - 1:])
        op = ekf_p.run(MU0, SIGMA0, e["u"], e["z"])
        est["raw"].append(rmse(raw_h[BURN:], e["x"][BURN:]))
        est["lat"].append(rmse(lat_h[BURN:], e["x"][BURN:]))
        est["phys"].append(rmse(op["mu"][BURN:], e["x"][BURN:]))
        starts = np.arange(BURN, T - max(horizons) - 1, 25)
        for k in horizons:
            for key, s0 in (("raw", lambda t: model.encode_target(Xe[t])[0]), ("lat", lambda t: o["mu"][t])):
                errs = []
                for t0_ in starts:
                    z = s0(t0_)
                    for j in range(1, k + 1):
                        z = model.predict(z, Ue[t0_ + j])
                    errs.append(readout(z)[0] - e["x"][t0_ + k])
                pred[key][k].append(np.sqrt(np.mean(np.array(errs) ** 2, 0)))
            errs = []
            for t0_ in starts:
                x = op["mu"][t0_].copy()
                for j in range(1, k + 1):
                    x = qt.g_disc(e["u"][t0_ + j], x, DT)
                errs.append(x - e["x"][t0_ + k])
            pred["phys"][k].append(np.sqrt(np.mean(np.array(errs) ** 2, 0)))
    RES["B_estimation_rmse"] = {k: np.mean(v, 0).round(4).tolist() for k, v in est.items()}
    RES["B_prediction_rmse"] = {key: {str(k): np.mean(v, 0).round(4).tolist() for k, v in d_.items()} for key, d_ in pred.items()}
    RES["B_latent_mean_nis"] = float(np.nanmean(np.array(lat_nis)[:, BURN:]))
    print(" B est", RES["B_estimation_rmse"])

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
    for key, lab, col in (("raw", "raw encoder", C["raw"]), ("lat", "latent EKF", C["lat"]), ("phys", "physics EKF + model", C["ekf"])):
        ax[0].plot(horizons, [np.mean(pred[key][k], 0)[0] for k in horizons], "o-", color=col, label=lab)
        ax[1].plot(horizons, [np.mean(pred[key][k], 0)[2] for k in horizons], "o-", color=col, label=lab)
    ax[0].set_title(r"$k$-step prediction RMSE, $h_1$"); ax[1].set_title(r"$k$-step prediction RMSE, $h_3$ (unmeasured)")
    for a in ax:
        a.set_xlabel("horizon $k$ (s)"); a.set_ylabel("cm")
    ax[0].legend(fontsize=7)
    save(fig, "b2_prediction")

    # example trajectory: unmeasured h3 from the three estimators
    e = qt.simulate(T=T, seed=TEST_SEEDS[0]); Xe = featurise([e], nf)[0]; Ue = nu(e["u"])
    o = run_latent_ekf(lekf, Xe, Ue, W_WIN - 1, T)
    op = ekf_p.run(MU0, SIGMA0, e["u"], e["z"])
    tt = np.arange(T)
    fig, ax = plt.subplots(2, 1, figsize=(7, 3.6), sharex=True)
    for j, i in enumerate([0, 2]):
        ax[j].plot(tt, e["x"][:, i], "k", label="true")
        ax[j].plot(tt[W_WIN - 1:], readout(model.encode_target(Xe[W_WIN - 1:]))[:, i], color=C["raw"], lw=.7, alpha=.8, label="raw encoder + readout")
        ax[j].plot(tt[W_WIN - 1:], readout(o["mu"][W_WIN - 1:])[:, i], color=C["lat"], label="latent EKF + readout")
        ax[j].plot(tt, op["mu"][:, i], color=C["ekf"], ls="--", label="physics EKF")
        ax[j].set_ylabel(f"$h_{i+1}$ (cm)")
    ax[0].legend(fontsize=7, ncol=2); ax[1].set_xlabel("time (s)"); ax[1].set_xlim(200, 1800)
    save(fig, "b3_latent_tracking")

    # ---- fault detection: physics EKF vs latent EKF (thresholds from fault-free validation runs)
    def lat_stat(ep):
        o = run_latent_ekf(lekf, featurise([ep], nf)[0], nu(ep["u"]), W_WIN - 1, T)
        return win_stat(o["nis"], D_LAT)
    def phys_stat(ep):
        return win_stat(physics_ekf().run(MU0, SIGMA0, ep["u"], ep["z"])["nis"], 2)
    K_DET = 30
    def kstep_stat(ep):
        """k-step latent residual ('imagination error'): ||f_bar(x_t) - p^k(f_bar(x_{t-k}), u)||^2, window-averaged."""
        Xe = featurise([ep], nf)[0]; Ue = nu(ep["u"])
        Y = np.full((T, D_LAT), np.nan); Y[W_WIN - 1:] = model.encode_target(Xe[W_WIN - 1:])
        ts = np.arange(W_WIN - 1 + K_DET, T)
        Zk = Y[ts - K_DET]
        for j in range(K_DET, 0, -1):
            Zk = model.predict(Zk, Ue[ts - j + 1])
        r = np.full(T, np.nan); r[ts] = np.sum((Y[ts] - Zk) ** 2, axis=1)
        x = np.nan_to_num(r, nan=float(np.nanmean(r)))
        return np.convolve(x, np.ones(WIN), mode="full")[:T]
    RES["B_latent_noise"] = dict(trace_R_lat=float(np.trace(R_lat)), trace_Q_lat=float(np.trace(Q_lat)))
    calib = [qt.simulate(T=T, seed=s) for s in range(300, 320)]
    thr = {}
    DETECTORS = (("physics", phys_stat), ("latent", lat_stat), ("latent_kstep", kstep_stat))
    for name, fn in DETECTORS:
        mx = [fn(ep)[WIN + BURN:].max() for ep in calib]
        thr[name] = float(np.quantile(mx, 0.95))          # <= 5% of fault-free runs raise any alarm
    det = {}
    examples = {}
    for fname, kw in FAULTS.items():
        det[fname] = {}
        for name, fn in DETECTORS:
            fas, delays = [], []
            for s in TEST_SEEDS[:30]:
                ep = qt.simulate(T=T, seed=s, **kw)
                st = fn(ep)
                fa, dl = detect(st, thr[name], FAULT_T)
                fas.append(fa); delays.append(dl)
                if s == TEST_SEEDS[0]:
                    examples[(fname, name)] = st
            ok = [x for x in delays if x is not None]
            det[fname][name] = dict(detection_rate=len(ok) / len(delays),
                                    median_delay_s=float(np.median(ok)) if ok else None,
                                    false_alarm_rate=float(np.mean(fas)))
    # detection rate versus leak size (how large must a fault be before each monitor sees it?)
    sizes = [0.01, 0.02, 0.04, 0.08]
    sweep = {name: [] for name, _ in DETECTORS}
    for a_leak in sizes:
        for name, fn in DETECTORS:
            hits = 0
            for s in TEST_SEEDS[:15]:
                ep = qt.simulate(T=T, seed=s, leak=(2, a_leak), leak_start=FAULT_T)
                hits += detect(fn(ep), thr[name], FAULT_T)[1] is not None
            sweep[name].append(hits / 15)
    RES["B_leak_sweep"] = dict(sizes_cm2=sizes, detection_rate=sweep)
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for name, _ in DETECTORS:
        ax.plot(sizes, sweep[name], "o-", label=name.replace("_", " "))
    ax.set_xscale("log", base=2); ax.set_xlabel(r"leak outlet area in tank 3 (cm$^2$)"); ax.set_ylabel("detection rate")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)
    save(fig, "b5_leak_sweep")
    RES["B_fault_detection"] = dict(thresholds=thr, results=det, window=WIN)
    print(" B faults", det)
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    titles = {"physics": "physics EKF: NIS", "latent": "latent EKF: NIS", "latent_kstep": f"latent model: {K_DET}-step residual"}
    cols = {"physics": C["ekf"], "latent": C["lat"], "latent_kstep": C["raw"]}
    for r, fname in enumerate(FAULTS):
        for c, (name, _) in enumerate(DETECTORS):
            st = examples[(fname, name)]
            ax[r, c].semilogy(st, color=cols[name])
            ax[r, c].axhline(thr[name], c="k", ls="--", lw=.8); ax[r, c].axvline(FAULT_T, c="r", lw=.8)
            ax[r, c].set_title(f"{fname.replace('_', ' ')}\n{titles[name]}", fontsize=7)
    for a in ax[1]:
        a.set_xlabel("time (s)")
    ax[0, 0].set_ylabel(f"window sum ({WIN} s)"); ax[1, 0].set_ylabel(f"window sum ({WIN} s)")
    save(fig, "b4_fault_detection")


if __name__ == "__main__":
    t0 = time.time()
    part_a()
    part_b(None)
    RES["runtime_seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "results.json").write_text(json.dumps(RES, indent=1))
    print("done in", RES["runtime_seconds"], "s")
