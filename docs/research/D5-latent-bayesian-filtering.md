# D5: Latent Bayesian filtering and innovation-based trust monitoring for decoder-free world models

**Status:** chosen direction (2026-11-14) · **Target:** Q1 journal · **Owner:** Nilesh Ravishan
**Seed result:** the EN4594 mini project (`report/` PDF, code in `src/lbf`, `experiments/`)

**Working title:** *Can I trust my world model? Bayesian filtering and innovation-based monitoring for
decoder-free (JEPA) world models of controlled industrial processes*

---

## 1. One-paragraph pitch
Latent world models such as JEPA predict *representations* of future observations, not the observations
themselves. They are becoming the standard way to learn dynamics from sensor logs with control inputs, but
they give no calibrated answer to two operational questions:
1. *What is the current state, given noisy sensors?*
2. *Is the model still right?*

We show how to run a **Bayesian filter (EKF/UKF, and a particle filter for multimodal cases) inside the
latent space of a decoder-free, action-conditioned JEPA**, using the target encoder's embedding of each new
sensor window as the measurement. We make the filter's **innovation statistic (NIS) calibrated** by learning an
input-dependent process-noise model and handling the correlated measurement noise. We then use it as a
**run-time trust and fault monitor with a statistically defined false-alarm rate**, and as the belief state
for **MPC**. We evaluate on a nonlinear benchmark (quadruple tank), the Tennessee Eastman Process, and a
**real multiphase-flow plant (PRONTO, CC BY 4.0)**.

## 2. Why this matters (link to the notebook's thesis)
- The source blog (`docs/sources/01`) names **industrial sensor streams with control inputs** as the
  domain-adaptation target. It lists open problems #2 (*error grows with rollout horizon*) and #4 (*latency and
  reliability for deployed systems*). A filter fixes drift at every step (#2), and a calibrated innovation test
  is a reliability monitor (#4).
- Direction D3 asks whether imagination error is a proxy for control quality. The NIS is the *principled,
  normalised* version of imagination error: it is χ²-distributed when the model is right (Mehra 1970;
  Piché 2016 shows the online NIS test equals a Bayesian prior-predictive p-test).
- Recent evidence: one-step validation error does **not** predict planning success in latent world models
  (rank corr. 0.024), while rollout fidelity does (0.62) (You et al., arXiv 2607.10362). So a *calibrated*
  online error signal is needed.

## 3. Closest prior work, and our gap (verified, see `docs/literature/latent-filtering-related-work.md`)
| Work | Filter in latent? | Decoder-free? | Actions/control? | Calibrated innovation monitor? | MPC? | Real industrial data? |
|---|---|---|---|---|---|---|
| KVAE (Fraccaro et al., NIPS 2017) | ✅ KF | ❌ VAE decoder | ✅ | ❌ | ❌ | ❌ |
| RKN (Becker et al., ICML 2019) | ✅ factorised KF | ❌ | ❌ | ❌ | ❌ | ❌ |
| DVBF (Karl et al., ICLR 2017) / E2C (Watter et al., NIPS 2015) | ✅ (variational / locally linear) | ❌ | ✅ | ❌ | E2C ✅ | ❌ |
| MTS3 (Shaj et al., NeurIPS 2023) | ✅ Kalman-style updates | ❌ | ✅ | ❌ | ❌ | ❌ |
| **K-PWM** (Rajkumar et al., NeurIPS 2026, arXiv 2611.11521) | ✅ EKF + smoother | ❌ nonlinear decoder | ✅ | ❌ | ✅ MPPI | ❌ (MuJoCo) |
| Deshpande et al. (IEEE L-CSS 2023) | ✅ KF on autoencoder latent | ❌ | ✅ | ❌ | ❌ | ✅ (vapour compression, physics-constrained) |
| Falconer et al. (Physica D 2025) | ✅ EnKF in a Koopman-AE latent | ❌ | ❌ | ❌ | ❌ | streams |
| Mejari et al. (arXiv 2607.14273) | ✅ differentiable KF, innovation-likelihood loss | ❌ | ✅ | conformal bounds | ❌ | ❌ |
| VJEPA (Huang, arXiv 2601.14354) | "analogous to" Bayesian filtering | ✅ | ✅ | ❌ | claimed | ❌ |
| AdaJEPA (Wang, LeCun et al., arXiv 2606.32026) | ❌ (test-time adaptation) | ✅ | ✅ | uses step error to adapt | ✅ | ❌ |
| CDR-LWM (Wang, *Electronics* 2026) | ❌ | ❌ | ✅ | conformal ensemble radius | ✅ robust MPC | ❌ |
| KalmanNet (Revach et al., IEEE TSP 2022) + Bayesian KalmanNet (Dahan et al., IEEE TSP 2025) | learned gain | n/a | partly | uncertainty via sampling | ❌ | ❌ |
| **Ours (D5)** | ✅ EKF/UKF/PF | ✅ | ✅ | ✅ calibrated NIS + CUSUM, controlled false-alarm rate | ✅ | ✅ TEP + PRONTO |

**Novelty claim, stated carefully:** our contribution is the *combination*:
- a decoder-free JEPA world model;
- Bayesian filtering in its latent space, with encoder embeddings as measurements;
- **calibrated** innovation-based trust monitoring with guaranteed false-alarm control;
- use of the monitor and belief in MPC;
- validation on real process data with labelled faults.

No single ingredient is new by itself. Say this explicitly in the paper.

## 4. Research questions and hypotheses
| RQ | Question | Hypothesis / success criterion |
|---|---|---|
| RQ1 | When does latent filtering beat using the raw encoder embedding? | Gain ∝ ratio of predictor error to embedding noise. Theory: steady-state gain ≈ R(R+Q)⁻¹ in 1-D. Make R_lat < Q_lat through better training (multi-step, larger K) and show ≥20 % RMSE reduction over the raw encoder |
| RQ2 | Can the latent NIS be calibrated? | With a heteroscedastic process noise R_lat(s,u) (predicted by a head on the predictor), and with correlated measurement noise handled (state augmentation or decorrelating), ANIS stays inside the χ² band in ≥90 % of windows, *including* input transients |
| RQ3 | Is calibrated latent NIS a good trust and fault monitor? | On TEP (Rieth/Reinartz) and PRONTO, it matches or beats PCA T²/SPE, autoencoder residuals, deep ensembles and conformal radius on detection delay at a fixed false-alarm rate. It also detects model drift (concept shift) |
| RQ4 | Does the filtered belief plus trust gating improve control? | On quadruple-tank and TEP closed loops, MPC from the filtered latent belief has lower cost and fewer constraint violations than MPC from raw embeddings; trust gating falls back to a safe controller when NIS alarms |

## 5. Method
### 5.1 World model (as in the seed code, scaled up in PyTorch)
**Components:**
- Encoder f_θ over a window w_t of [z, u].
- EMA target encoder f̄ (BYOL / I-JEPA style).
- Action-conditioned residual predictor p_φ(s, u).
- New: a **noise head** that outputs a diagonal or low-rank covariance R_φ(s, u).

**Loss:** multi-step latent prediction, plus VICReg anti-collapse, plus a **Gaussian NLL term** on the
k-step residuals so that R_φ is calibrated (β-NLL weighting to avoid variance collapse).

**Options:**
- Initialise the encoder from a time-series foundation model (MOMENT, Chronos). This tests the "what
  transfers?" question (D2).

### 5.2 Latent filter
Latent state-space model:
```
s_t = p_φ(s_{t-1}, u_t) + ε_t,     ε_t ~ N(0, R_φ(s_{t-1}, u_t))        (motion model g)
y_t = f̄(w_t) = s_t + δ_t,           δ_t ~ N(0, Q_lat)                     (h(s) = s, H = I)
```

**Filter variants:**
- EKF: Jacobian G_t via autograd.
- UKF: no Jacobians (Julier & Uhlmann 2004).
- Particle filter: for multimodal or regime-switching cases.

**Correlated measurement noise:** overlapping windows make δ_t autocorrelated (lag-1 ≈ 0.27 in the seed).
Three remedies:
1. Augment the state with a first-order noise model (Bryson–Henrikson style).
2. Use non-overlapping or strided windows.
3. Learn the encoder with a decorrelation penalty across time.

**Noise estimation without simulation oracles:**
- Innovation-based adaptive estimation (Mehra 1970).
- Variational-Bayes adaptive KF (Huang et al. 2018 IEEE TAC).

### 5.3 Trust and fault monitor
- **Statistics:**
  - NIS_t ~ χ²_d under H0;
  - windowed sum, ~ χ²_{Ld};
  - CUSUM / GLR on NIS (Basseville & Nikiforov 1993).
- **Threshold:** set analytically from χ² when calibrated, otherwise by conformal calibration on fault-free
  data. The goal is a **controlled false-alarm rate**.
- **Diagnosis:** per-dimension innovations, mapped back through a readout or attribution, to say *which*
  sensor or subsystem is suspicious.

### 5.4 Control
- Latent MPC: CEM or MPPI over p_φ, starting from the filtered belief, with uncertainty-aware cost.
- Trust gating: fall back to a safe controller or a human operator when the monitor alarms.

## 6. Experiments
| ID | System | Data | Purpose |
|---|---|---|---|
| E1 | Quadruple tank (Johansson 2000) | Simulated (seed code) | RQ1–RQ2 ablations; fault sweep (leak size, sensor bias, actuator degradation) |
| E2 | Tennessee Eastman | Rieth et al. 2017 (Harvard Dataverse; 41 XMEAS + **11 XMV**, 20 faults, 500 runs per fault); Reinartz et al. 2021 (28 faults, 6 modes) | RQ3: standard FDD benchmark; compare against the Yin et al. 2012 baselines |
| E3 | **Real** multiphase-flow rig | **PRONTO** (Stief et al., J. Process Control 2019; Zenodo 10.5281/zenodo.1341583, CC BY 4.0; 1 Hz process data, valve positions, setpoints varied; faults: air leak, air blockage, diverted flow, slugging) | RQ3 on real data. **First check that setpoints and valve commands are separate channels** |
| E4 | Quadruple tank + TEP closed loop | Simulated | RQ4 MPC with and without filtering, with and without gating |
| E5 | Optional: LBNL HVAC fault data (Granderson et al., Sci. Data 2023; mostly simulated) or SWaT (attack labels, access by request) | — | Generality |

**Baselines:**
- Physics EKF (where a model exists).
- PCA T²/SPE (Qin 2003).
- Autoencoder or LSTM residuals (Lomov et al. 2021).
- Deep ensembles (Lakshminarayanan et al. 2017).
- Conformal radius (CDR-LWM style).
- KalmanNet and Bayesian KalmanNet.
- KVAE or RKN-style latent KF (decoder-based).
- RSSM / Dreamer-style filter (PlaNet).
- Raw-encoder JEPA without a filter.

**Metrics:**
- Estimation: RMSE (where ground truth exists).
- Calibration: ANEES/ANIS coverage, expected calibration error.
- Detection: delay at a fixed false-alarm rate, detection rate, AUROC, missed-alarm rate.
- Control: closed-loop cost, constraint violations.
- Cost: compute per step.

**Statistics:** ≥5 seeds; paired tests; bootstrap confidence intervals.

## 7. Seed results already obtained (EN4594 mini project, 2026-11-14)
- **Physics EKF on the quadruple tank** (50 Monte Carlo runs):
  - RMSE: 0.038 cm on the measured levels, 0.034–0.038 cm on the unmeasured levels.
  - Consistent: ANEES 4.10, ANIS 2.01, inside their χ² bands 95 % of the time.
  - NIS monitor: detects a 0.02 cm² leak in the hidden tank (median delay 36 s) and a 0.5 cm sensor bias (5 s), with 0 false alarms.
- **NumPy JEPA (d = 12) + latent EKF:**
  - Recovers the unmeasured h₃ to 0.107 cm with no physics.
  - **But** tr R_lat = 0.072 > tr Q_lat = 0.023, so filtering gains only about 2 % over the raw encoder.
  - The latent NIS spikes at input steps. As a result it detects the 0.02 cm² leak in 23 % of runs (vs 100 % for physics), and reaches 87 % only at 0.08 cm².
- **This motivates RQ1 and RQ2.** The bottleneck is the calibration of the world model's uncertainty, not the filter.

## 8. Plan and timeline
| Period | Work | Output |
|---|---|---|
| until ~20 Nov 2026 | Submit the EN4594 report (done; see `report/`) | Coursework |
| late Nov–Dec 2026 | Port to PyTorch; noise head + NLL; correlated-noise handling; UKF; adaptive Q/R; E1 ablations | RQ1–RQ2 answered on E1 |
| Dec 2026–early Jan 2027 | TEP pipeline (Rieth data); baselines (PCA, ensembles, KalmanNet); E2 | First RQ3 table |
| by 9 Jan 2027 | **Short arXiv preprint / workshop paper** (E1 + first E2): useful for the Imperial application (deadline 11 Jan) and HKPFS interviews | Preprint |
| Feb–Mar 2027 | PRONTO real data (E3); MPC + gating (E4) | Full results |
| Apr–May 2027 | Write the journal paper; internal review (ask Monash and CRISiSLab supervisors for feedback) | Submission |

## 9. Target journals (fees verified, see `docs/publishing/target-journals.md`)
1. **Journal of Process Control** (Elsevier):
   - fit: process FDD and control;
   - cost: no fee on the subscription route;
   - length: ≤12 double-column pages.
2. **Engineering Applications of Artificial Intelligence** (Elsevier):
   - fit: broad AI + engineering;
   - cost: free on the subscription route.
3. **IEEE TNNLS:**
   - fit: if the method and theory are strong;
   - cost: free up to 10 pages, then $200 per page (mandatory).
4. **IEEE Trans. Industrial Informatics:**
   - cost: 8 pages, then mandatory fees.

Check quartiles on SJR before choosing.

## 10. Risks
| Risk | Mitigation |
|---|---|
| Fast-moving area; someone publishes the same combination | Post an arXiv preprint early (by Jan 2027); keep the novelty claim tied to the *combination + real data + calibration* |
| The latent NIS cannot be calibrated through transients | That is a publishable negative result (connect it to the D3 proxy question); fall back to a conformal threshold |
| Compute | Models are small (latents of 8–32 dims; MLP/TCN encoders). Free Colab or Kaggle GPUs are enough. The seed trains on a CPU in 35 s |
| PRONTO lacks clean control channels | Use valve positions as actions; otherwise use Granderson HVAC (control signals logged) |
| Reviewers ask "why not physics?" | Physics-EKF upper bound is reported. The method targets plants without a usable first-principles model |
