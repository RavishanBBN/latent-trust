# CLAUDE.md: World Models research notebook + D5 latent-filtering project (read this first)

You are helping **Nilesh Ravishan** turn this project into a **Q1 journal paper**. He is:
- a final-year Electronic & Telecommunication Engineering student at the University of Moratuwa, Sri Lanka, graduating June 2027;
- a research assistant at Monash University's Exertion Games Lab and at CRISiSLab, Massey University;
- applying for funded PhDs (HKPFS deadline 1 Dec 2026; Imperial DEPS deadline 11 Jan 2027).

A strong preprint before January 2027 helps those applications. Budget is limited, so he targets **hybrid journals published for free** through the subscription route.

## 1. What this repo is
It is a research notebook (started 2026-10-13) that is now also a code repo:
- `docs/`: the notebook. Start with `docs/PROGRESS.md`, then the chosen direction
  **`docs/research/D5-latent-bayesian-filtering.md`** (the full Q1 plan).
- `src/lbf/`: NumPy package. It contains:
  - `quadtank.py`: quadruple-tank plant and Jacobians;
  - `ekf.py`: EKF in course notation;
  - `metrics.py`: NEES/NIS, χ² bands, CUSUM;
  - `autodiff.py`: a tiny reverse-mode autodiff, written because torch was unavailable;
  - `jepa.py`: an action-conditioned JEPA world model;
  - `latent_ekf.py`: an EKF in the JEPA latent space.
- `experiments/run_assignment.py`: reproduces every number and figure of the EN4594 report. It takes about 7 min on a CPU and writes `results/`.
- `assignment/`: the submitted EN4594 mini-project report (`report.tex`, `report.pdf`).
- `tests/test_autodiff.py`: a finite-difference gradient check.

## 2. The research idea (D5) in one paragraph
Run a Bayesian filter (EKF, UKF or particle filter) **inside the latent space of a decoder-free, action-conditioned JEPA world model**, treating the target encoder's embedding of each new sensor window as the measurement (h(s) = s). Then:
1. make the filter's **innovation statistic (NIS) calibrated**, using a learned, input-dependent process noise R(s, u) and by handling correlated measurement noise;
2. use it as a **run-time "can I trust my world model?" / fault monitor** with a controlled false-alarm rate;
3. use the filtered belief for **MPC**.

Evaluate on the quadruple tank, the Tennessee Eastman Process and the **real PRONTO multiphase-flow dataset (CC BY 4.0)**. Closest prior work is K-PWM (NeurIPS 2026: latent EKF + MPPI, but decoder-based, no monitor), MTS3, KVAE, RKN and the KalmanNet family. The novelty is the *combination* (see D5 §3).

## 3. Status (2026-11-14)
**Physics EKF (50 Monte Carlo runs):**
- RMSE 0.038 cm on the measured levels and 0.034–0.038 cm on the unmeasured ones.
- Consistent: ANEES 4.10, ANIS 2.01, inside their χ² bands about 95 % of the time.
- The NIS monitor detects a hidden-tank leak (36 s) and a sensor bias (5 s), with 0 false alarms.

**JEPA (d = 12) + latent EKF:**
- Recovers the unmeasured h₃ to 0.107 cm without physics.
- But **tr R_lat = 0.072 > tr Q_lat = 0.023**, so filtering adds only about 2 % over the raw encoder.
- The latent NIS spikes at input steps, so leak detection is only 23 % at 0.02 cm² and 87 % at 0.08 cm².

**This is the starting point:** fix the world model's uncertainty calibration (RQ1–RQ2 in D5).

## 4. Rules (inherited from the notebook; keep them)
1. **Every session:** add a dated entry to `docs/PROGRESS.md` (Done / Findings / Decisions / Next), add a line under `## [Unreleased]` in `CHANGELOG.md`, and commit with a short message. Tag releases `v0.x.0` at milestones.
2. **Never record a paper claim from memory.** Every paper needs a link and a verification level:
   - `search`: only the search result was seen;
   - `abstract`: the abstract was read;
   - `full`: the full paper was read;
   - `code`: its code was run.

   Literature lives in `docs/literature/`. The SSRN Lahoud paper and K-PWM must be read in full before claiming novelty.
3. **Honest evaluation:**
   - fixed seeds, at least 5 training seeds per configuration;
   - report mean ± CI;
   - choose thresholds on fault-free validation data only;
   - never tune on test faults;
   - report negative results (they are informative here).
4. **Keep the physics EKF as an upper-bound reference** in every experiment where a model exists.
5. **Data** goes in `data/` (git-ignored). Write download scripts; respect dataset licences (PRONTO CC BY 4.0, Rieth TEP via Harvard Dataverse, SWaT on request only).
6. **Uncommitted work:** the files added on 2026-11-14 (CLAUDE.md, D5, literature, code, assignment) are **not committed yet**. Review them, then run `git add -A && git commit -m "D5 direction, EN4594 seed code and report"` and `git tag v0.3.0`.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_autodiff.py             # gradient check
python experiments/run_assignment.py      # ~7 min CPU, writes results/figures + results/results.json
```

## 6. Next tasks (in order; see D5 §8 for dates)
1. Commit (rule 6). Add `pytest` tests:
   - EKF on a linear system must equal the KF;
   - the Jacobian must match finite differences (`predict_jacobian` and `quadtank.G_disc`);
   - `metrics.chi2_band`.
2. **Port the world model to PyTorch** (`src/lbf_torch/` or a refactor), keeping the NumPy version for the assignment's reproducibility. Use autograd Jacobians for the EKF.
3. **RQ1/RQ2 on the quadruple tank:**
   - noise head R_φ(s, u) with β-NLL on k-step residuals;
   - larger K, longer training, latent dimension sweep (8–32);
   - strided windows or noise-state augmentation for correlated δ;
   - adaptive Q/R (Mehra 1970; Huang et al. 2018);
   - UKF variant.

   Target: ANIS inside the χ² band through input transients, and ≥20 % RMSE gain over the raw encoder.
4. **TEP pipeline** (Rieth et al. 2017 data) with baselines: PCA T²/SPE, autoencoder residuals, deep ensembles and KalmanNet. Report detection delay at a fixed false-alarm rate.
5. **Short arXiv preprint by about 9 Jan 2027** (E1 + first E2).
6. **PRONTO real data** (check whether setpoints and valve commands are separate channels), then MPC with trust gating.
7. Write the journal paper (J. Process Control or EAAI first; IEEE TNNLS at ≤10 pages as a stretch).

## 7. Notation (match the EN4594 course and *Probabilistic Robotics*)
- Model: x_t = g(u_t, x_{t-1}) + ε_t with ε_t ~ N(0, **R**) (process noise); z_t = h(x_t) + δ_t with δ_t ~ N(0, **Q**) (measurement noise).
- Jacobians G_t and H_t; Joseph-form covariance update.
- Note: many ML papers swap the roles of Q and R, so always state the convention.
