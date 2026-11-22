# Related work for D5: latent Bayesian filtering, learned filters, monitoring (verified 2026-11-14)

How these were checked: each entry was looked up on a primary page, either the publisher, the DOI landing page, Crossref metadata, proceedings, or the arXiv abstract. These are `abstract`-level verifications, which is the notebook's convention. Upgrade an entry to `full` once the paper has been read. Earlier JEPA and world-model papers are already in `docs/reading-list.md` and `docs/literature/jepa-sensor-streams.md`.

## A. Filtering foundations

| Key | Citation | Link | Checked | Use in D5 |
|---|---|---|---|---|
| kalman1960 | Kalman (1960). A New Approach to Linear Filtering and Prediction Problems. *J. Basic Eng.* 82(1):35–45 | https://doi.org/10.1115/1.3662552 | Crossref | Base filter |
| julier2004 | Julier & Uhlmann (2004). Unscented Filtering and Nonlinear Estimation. *Proc. IEEE* 92(3):401–422 | https://doi.org/10.1109/JPROC.2003.823141 | IEEE Xplore | UKF variant (no Jacobians) |
| thrun2005 | Thrun, Burgard, Fox (2005). *Probabilistic Robotics*. MIT Press | https://mitpress.mit.edu/9780262201629/probabilistic-robotics/ | MIT Press | Notation (R = process, Q = measurement) |
| sarkka2013/2023 | Särkkä (2013); Särkkä & Svensson (2023, 2nd ed.). *Bayesian Filtering and Smoothing*. CUP | https://doi.org/10.1017/CBO9781139344203 · https://doi.org/10.1017/9781108917407 | Crossref | Theory |
| barshalom2001 | Bar-Shalom, Li, Kirubarajan (2001). *Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software*. Wiley | https://doi.org/10.1002/0471221279 | Crossref | NEES/NIS tests (**confirm the exact section before citing**) |
| mehra1970 | Mehra (1970). On the identification of variances and adaptive Kalman filtering. *IEEE TAC* 15(2):175–184 | https://doi.org/10.1109/TAC.1970.1099422 | Crossref | Innovation-based optimality tests; adaptive Q/R |
| huang2018 | Huang, Zhang, Wu, Li, Chambers (2018). A Novel Adaptive Kalman Filter With Inaccurate Process and Measurement Noise Covariance Matrices. *IEEE TAC* 63(2):594–601 | https://doi.org/10.1109/TAC.2017.2730480 | Crossref | VB adaptive Q/R baseline |
| piche2016 | Piché (2016). Online tests of Kalman filter consistency. *Int. J. Adapt. Control Signal Process.* 30(1):115–124 | https://doi.org/10.1002/acs.2571 | Tampere portal | **The NIS test is a Bayesian prior-predictive p-test** (theory for the monitor) |
| gordon1993 | Gordon, Salmond, Smith (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proc. F* 140(2):107ff | https://doi.org/10.1049/ip-f-2.1993.0015 | Crossref | Particle filter |
| arulampalam2002 | Arulampalam, Maskell, Gordon, Clapp (2002). A tutorial on particle filters… *IEEE TSP* 50(2):174–188 | https://doi.org/10.1109/78.978374 | Crossref | Particle filter |
| willsky1976 | Willsky (1976). A survey of design methods for failure detection in dynamic systems. *Automatica* 12(6):601–611 | https://doi.org/10.1016/0005-1098(76)90041-8 | Crossref | Residual-based fault detection |
| basseville1993 | Basseville & Nikiforov (1993). *Detection of Abrupt Changes: Theory and Application*. Prentice Hall | https://people.irisa.fr/Michele.Basseville/kniga/kniga.html | author page | CUSUM/GLR on NIS |
| gama2014 | Gama et al. (2014). A survey on concept drift adaptation. *ACM CSUR* 46(4):1–37 | https://doi.org/10.1145/2523813 | Crossref | Drift framing |

## B. Learned and latent filters (closest families)

| Key | Citation | Link | Checked | Relation to D5 |
|---|---|---|---|---|
| haarnoja2016 | Haarnoja, Ajay, Levine, Abbeel (2016). Backprop KF. *NIPS 29* | https://arxiv.org/abs/1605.07148 | NeurIPS page + arXiv | KF over learned features (trained end to end) |
| krishnan2017 | Krishnan, Shalit, Sontag (2017). Structured Inference Networks for Nonlinear State Space Models. *AAAI* 31(1) | https://doi.org/10.1609/aaai.v31i1.10779 | AAAI page | Deep state-space models (amortised); "Deep Kalman Filters" (arXiv 1511.05121) is a preprint only |
| fraccaro2017 | Fraccaro, Kamronn, Paquet, Winther (2017). A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning (KVAE). *NIPS 30* | https://proceedings.neurips.cc/paper/2017/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html | NeurIPS page | **Exact KF in a latent space, but with a decoder** |
| karl2017 | Karl, Soelch, Bayer, van der Smagt (2017). Deep Variational Bayes Filters. *ICLR* | https://arxiv.org/abs/1605.06432 | ICLR archive + arXiv | Variational latent filter with control |
| becker2019 | Becker et al. (2019). Recurrent Kalman Networks. *ICML*, PMLR 97:544–552 | https://proceedings.mlr.press/v97/becker19a.html | PMLR | Factorised latent KF |
| watter2015 | Watter et al. (2015). Embed to Control. *NIPS 28* | https://arxiv.org/abs/1506.07365 | NeurIPS page | Locally linear latent dynamics for control |
| lusch2018 | Lusch, Kutz, Brunton (2018). Deep learning for universal linear embeddings of nonlinear dynamics. *Nat. Commun.* 9:4950 | https://doi.org/10.1038/s41467-018-07210-0 | Crossref | Koopman latent (near-linear allows an exact KF) |
| revach2022 | Revach et al. (2022). KalmanNet. *IEEE TSP* 70:1532–1547 | https://doi.org/10.1109/TSP.2022.3158588 | Crossref | Learned gain; no explicit covariance (baseline) |
| klein2022 | Klein et al. (2022). Uncertainty in Data-Driven Kalman Filtering for Partially Known State-Space Models. *ICASSP*, 3194–3198 | https://doi.org/10.1109/ICASSP43922.2022.9746732 | Crossref | Covariance for KalmanNet |
| dahan2025 | Dahan, Revach, Duník, Shlezinger (2025). Bayesian KalmanNet. *IEEE TSP* 73:2558–2573 | https://doi.org/10.1109/TSP.2025.3581703 | Crossref | Uncertainty-aware learned KF baseline |
| choi2023 | Choi et al. (2023). Split-KalmanNet. *IEEE TVT* 72(9):12326–12331 | https://doi.org/10.1109/TVT.2023.3270353 | Crossref | Separates process and measurement mismatch |
| shlezinger2025 | Shlezinger et al. (2025). AI-Aided Kalman Filters. *IEEE SPM* 42(3):52–76 | https://doi.org/10.1109/MSP.2025.3569395 | Crossref | **Survey to position against** |
| shaj2022 | Shaj et al. (2022). Hidden Parameter Recurrent State Space Models for Changing Dynamics Scenarios. *ICLR* | https://arxiv.org/abs/2206.14697 | ICLR page | Latent filtering under changing dynamics |
| shaj2023 | Shaj Kumar et al. (2023). Multi Time Scale World Models (MTS3). *NeurIPS 36* | https://proceedings.neurips.cc/paper_files/paper/2023/hash/54d8aab579b5a9ed3395764c7341ebec-Abstract.html | NeurIPS + PDF | **Kalman updates in an action-conditioned latent world model** (no JEPA, no monitoring) |
| deshpande2023 | Deshpande, Chakrabarty, Vinod, Laughman (2023). Physics-Constrained Deep Autoencoded Kalman Filters for Estimating Vapor Compression System States. *IEEE L-CSS* 7:3483–3488 | https://doi.org/10.1109/LCSYS.2023.3334959 | MERL TR + Crossref | Control-journal precedent: latent KF on a real system |
| falconer2025 | Falconer, Lloyd, Santitissadeekorn (2025). Tracking and forecasting oscillatory data streams using Koopman autoencoders and Kalman filtering. *Physica D* 476:134649 | https://doi.org/10.1016/j.physd.2025.134649 | Crossref | EnKF in a learned latent space on streams |
| becker2022 | Becker & Neumann (2022). On Uncertainty in Deep State Space Models for Model-Based RL. *TMLR* | https://openreview.net/forum?id=UQXdQyoRZh | KIT page only | Uncertainty in latent state-space models |

## C. World models, uncertainty and control

| Key | Citation | Link | Checked | Relation |
|---|---|---|---|---|
| hafner2019 | Hafner et al. (2019). PlaNet. *ICML*, PMLR 97:2555–2565 | https://proceedings.mlr.press/v97/hafner19a.html | PMLR | RSSM baseline |
| chua2018 | Chua et al. (2018). PETS. *NeurIPS 31* | https://arxiv.org/abs/1805.12114 | NeurIPS page | Probabilistic ensembles + MPC |
| lakshminarayanan2017 | Lakshminarayanan, Pritzel, Blundell (2017). Deep Ensembles. *NIPS 30* | https://arxiv.org/abs/1612.01474 | NeurIPS page | Trust-signal baseline |
| kuleshov2018 | Kuleshov, Fenner, Ermon (2018). Calibrated Regression. *ICML*, PMLR 80:2796–2804 | https://proceedings.mlr.press/v80/kuleshov18a.html | PMLR | Recalibration of Q/R |
| wu2019 | Wu, Tran, Rincon, Christofides (2019). Machine learning-based predictive control of nonlinear processes. Part I: Theory. *AIChE J.* 65:e16729 | https://doi.org/10.1002/aic.16729 | author PDF | ML-based MPC for processes |

## D. 2025–2026 close works (mostly preprints; re-check status before submission)

| Work | Status | Link | Relation |
|---|---|---|---|
| **K-PWM**: structured bilinear world models under partial observation (Rajkumar et al.) | arXiv 2611.11521; **accepted at NeurIPS 2026** | https://arxiv.org/abs/2611.11521 | **Closest.** EKF filtering and smoothing in latent space plus MPPI; has a decoder, no innovation monitor, MuJoCo only |
| VJEPA (Huang) | arXiv 2601.14354 (Jan 2026) | https://arxiv.org/abs/2601.14354 | Probabilistic JEPA framed as Bayesian filtering; no sensor data or NIS |
| UWM-JEPA (Radha & Goktas) | arXiv 2605.25313 | https://arxiv.org/abs/2605.25313 | Belief space as a density matrix; no filter or MPC |
| AdaJEPA (Wang, LeCun et al.) | arXiv 2606.32026 | https://arxiv.org/abs/2606.32026 | Test-time adaptation driven by prediction error inside MPC |
| Control theory of predictability in latent world models (You et al.) | arXiv 2607.10362 | https://arxiv.org/pdf/2607.10362 | One-step error doesn't predict planning; rollout fidelity does |
| CDR-LWM (Wang) | *Electronics* 15(20):4765 (MDPI, 2026) | https://doi.org/10.3390/electronics15204765 | Conformal trust radius + robust MPC (decoder-based) |
| Reduced-order latent linear models for KF (Mejari, Dehkordi, Piga) | arXiv 2607.14273 | https://pith.science/paper/2607.14273 | Differentiable KF with innovation loss; heat-diffusion example |
| Kalman World Models (Kiruluta) | arXiv 2603.13423 | https://arxiv.org/html/2603.13423 | Kalman-style latent learning; no NIS, actions or MPC |
| FilterFM (Fan et al.) | arXiv 2610.40437 | https://arxiv.org/abs/2610.40437 | Pretrained amortised filter |
| Generative Bayesian Filtering (Cao et al.) | arXiv 2607.20521 | https://arxiv.org/abs/2607.20521 | CVAE likelihood filter; manufacturing data |
| Online monitoring of automotive time series with JEPA embeddings (Fertig et al.) | arXiv 2602.09985 | https://arxiv.org/html/2602.09985v1 | JEPA embeddings for monitoring; no filter |
| BeliefGraph-JEPA (Li et al.) | arXiv 2610.05409 | https://arxiv.org/abs/2610.05409 | Action-conditioned time-series JEPA (industrial); forecasting only |
| Ensemble KF in a VAE latent space (Pasmans et al.) | arXiv 2502.12987 | https://arxiv.org/abs/2502.12987 | Latent EnKF (geoscience) |
| JEPA latent world model for MPC (Santos) | Zenodo 10.5281/zenodo.23890647; **JEPA results "pending"** | https://zenodo.org/records/23890647 | Weak; do not cite as evidence |
| Lahoud: Kalman-inspired latent-state estimation, V-JEPA 2 for satellite AOI monitoring | SSRN 6521219; **abstract unverified** | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6521219 | Possibly close; **must read before claiming novelty** |

## E. JEPA / SSL ingredients used in the seed model
Assran et al. 2023 (I-JEPA, CVPR); Assran et al. 2025 (V-JEPA 2, arXiv 2506.09985); Grill et al. 2020 (BYOL,
NeurIPS); Bardes, Ponce, LeCun 2022 (VICReg, ICLR). Links for all four are in `docs/reading-list.md`.
