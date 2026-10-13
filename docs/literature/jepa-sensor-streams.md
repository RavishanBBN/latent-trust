# Related Work: JEPA and Latent World Models on Sensors and Time Series

Last updated: 2026-10-13 (venues verified)

**Verification levels:**
- `search`: only the title or snippet from a search result.
- `abstract`: the arXiv abstract page was read.
- `full`: the full paper was read.
- `code`: the code was run.

**Columns:**
- **AC?** = action- or control-conditioned (does the model take actions or control inputs?).
- **Adapt?** = does it reuse a pretrained general encoder and adapt it to the domain?

## A. Foundations: latent world models (mostly vision and robotics)

| Paper | Date | Domain | AC? | Adapt? | Key point | Code | Verified |
|-------|------|--------|-----|--------|-----------|------|----------|
| [V-JEPA 2 / V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) (Assran et al., Meta) | 2025-06 | Video → robot arm | ✅ | ✅ frozen video encoder | AC predictor post-trained on **<62 h** of unlabelled DROID robot video. Zero-shot pick-and-place on Franka arms via planning to image goals. | page lists none | abstract |
| [RoboJEPA](https://arxiv.org/abs/2610.10515) (Zholus et al.) | 2026-10 | 12 robot embodiments | ✅ | ✅ frozen V-JEPA encoder | 8B predictor. **Imagination error follows a power law in compute** and correlates with real-robot planning success, so it works as a cheap eval proxy. | promised (checkpoints + training + deploy code) | abstract |
| [D-JEPA](https://arxiv.org/abs/2609.24749) (Liu et al.) | 2026-09 | Robot manipulation, driving | ✅ | ? | Accurate prediction ≠ good action ranking, so it learns a **decision-aligned** latent geometry from ordinal evidence. 87.89% on PushT, +15.04 points on RoboTwin. | [project page](https://nebulis-lab.com/D-JEPA) | abstract |
| [JEPA-Anything](https://arxiv.org/abs/2609.20800) (Cui et al.) | 2026-09 | 7 domains: vision, biology, clinical trajectories, control, molecular dynamics, physical fields, weather | partly (control domain) | shared design, per-domain training | Orthogonal predictive factorization (OPF): latent targets split into complementary factors. −34.8% error on Interventional Pong. | [GitHub](https://github.com/Gen-Verse/JEPA-Anything) | abstract |

## B. Non-video sensors

| Paper | Date | Domain | AC? | Adapt? | Key point | Code | Verified |
|-------|------|--------|-----|--------|-----------|------|----------|
| [AD-LiST-JEPA](https://arxiv.org/abs/2602.12540) (Zhu & Choromanska) | 2026-02 | LiDAR (autonomous driving) | ✗ | ✗ (SSL pretrain, then downstream) | JEPA world model on unlabelled LiDAR. Evaluated on occupancy completion and forecasting. | none listed | abstract |

## C. Time series and industrial sensor streams ← closest to our direction

| Paper | Date | Domain | AC? | Adapt? | Key point | Code | Verified |
|-------|------|--------|-----|--------|-----------|------|----------|
| [Learning Actionable World Models for Industrial Process Control](https://arxiv.org/abs/2503.01411) (Yan et al., IEEE SDS 2025, Swiss Conf. on Data Science) | 2025-03 | **Plastic injection moulding** | ✅ factored process parameters | ✗ | JEPA-inspired with contrastive learning. Disentangles process parameters in latent space for fine-grained control, from very limited data. **The closest prior work.** | none listed | abstract |
| [TS-JEPA for Predictive Remote Control](https://arxiv.org/abs/2406.04853) (Girgis, Valcarce, Bennis; **IEEE IoT Journal 2026**) | 2024-06 | Networked control (inverted cart-pole) | ✅ (semantic actor) | ✗ | TS-JEPA with a semantic actor, for control under limited uplink capacity. | ? | search |
| [SC-JEPA (v1 title: MTS-JEPA)](https://arxiv.org/abs/2602.04643) (He et al., SDM 2026, to appear Nov 2026) | 2026-02 (v2 2026-07) | Multivariate TS anomaly **prediction** | ✗ | ✗ | Stabilises JEPA self-distillation on time series with a soft codebook bottleneck and a multi-resolution objective. Early warning on 5 benchmarks. | none listed | abstract |
| [CF-JEPA](https://arxiv.org/abs/2606.07031) (Lee & Sim) | 2026-06 | TS representation learning | ✗ | ✗ | Mask-free multi-horizon forward prediction. The online encoder is better for classification, the EMA target encoder for forecasting and anomaly detection. −27% forecasting MSE. Evaluated on UCR/UEA, ETT, KPI/Yahoo. | none listed | abstract |
| [Phys-JEPA](https://arxiv.org/abs/2606.16076) (Nie et al.) | 2026-06 | Multivariate TS forecasting | ✗ | ✗ | Latent state split into **physical + residual** parts, with physics consistency enforced on latents. Evaluated on Jena Climate, Traffic, Electricity. | none listed | abstract |

## D. Leads not yet checked (`search` only)

| Title | Link | Why it may matter |
|-------|------|-------------------|
| Hierarchical SSL Framework for Multivariate TS grounded in ECG (H-JEPA) | https://arxiv.org/abs/2607.01145 | Separate channel and temporal JEPA modules, relevant to multi-channel sensors |
| Hamiltonian JEPA: Action-Conditioned World Models with an Inherited Control State | https://arxiv.org/abs/2609.33497 | Separates the perception embedding from the control state |
| Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency | https://arxiv.org/abs/2608.12939 | Evaluation methodology for AC world models |
| Decision-Metric Alignment in Latent World Models (MPC planning) | https://arxiv.org/abs/2608.18746 | Latent world models used with MPC, which fits process control |
| Joint Embeddings Go Temporal (Ennadir, Golkar, Sarra), NeurIPS 2024 TSALM workshop | https://arxiv.org/abs/2509.25449 | The "TS-JEPA" other papers cite. Not the same paper as Girgis et al. |

## Observations (as of 2026-10-13)

1. **"JEPA on time series" alone is already crowded.** In 2026 there are at least four JEPA papers on
   time series (SC-JEPA, CF-JEPA, Phys-JEPA, H-JEPA). These papers do **forecasting, anomaly
   detection and classification**, which are passive prediction tasks.
2. **Action-conditioned JEPA on industrial sensor streams is thin.** The clearest example is Yan et al.
   (2025), which covers one proprietary process (injection moulding), shows no sign of public code or
   data, and starts from scratch with no general-encoder reuse.
3. **The V-JEPA 2-AC / RoboJEPA recipe (frozen general encoder + AC predictor + planning) has not been
   shown on industrial sensor streams**, as far as this survey found. This matches the blog's
   "domain adaptation" gap exactly.
4. RoboJEPA's result that **imagination error is a proxy for downstream control** is an untested
   hypothesis for process control, which makes it a clean, falsifiable research question.

## To do
- [ ] Read in full: RoboJEPA, Yan et al. 2025, JEPA-Anything (control domain section).
- [ ] Verify the D-section leads (open their abstract pages).
- [ ] Search for public **industrial datasets with actuator or control logs** (e.g., Tennessee Eastman
      Process simulator, SWaT/WADI water treatment, HVAC/building control). Check licences and whether
      the control inputs are actually recorded.
- [ ] Check which general **time-series foundation models** could serve as a frozen encoder (the
      analogue of V-JEPA in V-JEPA 2-AC).
