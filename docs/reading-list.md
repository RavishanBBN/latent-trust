# Reading List: Self-Supervised Learning → JEPA → World Models → Sensor Streams

Last updated: 2026-10-13

**How the venues were checked:**
- All venues were checked on 2026-10-13 against primary sources: arXiv metadata, PMLR, NeurIPS, CVF, OpenReview, Nature, IEEE, AAAI and IJCAI.
- Prefer the **arXiv** link for reading. It is free and usually the latest version.
- Use the **official** link when citing.

**Markers:**
- ★ = core path. Read these first.
- ☐ in the "Read" column: change it to ☑ when done, and record notes in `docs/literature/`.

**Venue guide:**

| Venue | What it is |
|-------|------------|
| ICML, NeurIPS, ICLR | Top machine-learning conferences |
| CVPR, ICCV | Top computer-vision conferences |
| Nature | Top science journal |
| TMLR | Peer-reviewed ML journal run by the ICLR/NeurIPS community |
| KDD, AAAI, IJCAI | Top data-mining / AI conferences |
| IEEE IoT Journal | Strong IEEE journal |
| SDM | Solid SIAM data-mining conference |
| IEEE SDS | Smaller regional IEEE conference |
| NeurIPS workshop | Lighter review than the main conference |

---

## Stage 1: Self-supervised learning (learning without labels)

| # | ★ | Paper | Venue | Why read it | Links | Read |
|---|---|-------|-------|-------------|-------|------|
| 1 | ★ | **SimCLR**: A Simple Framework for Contrastive Learning of Visual Representations (Chen et al.) | ICML 2020 | The clearest intro to contrastive learning: pull two views of the same thing together, push different things apart | [arXiv](https://arxiv.org/abs/2002.05709) · [official](https://proceedings.mlr.press/v119/chen20j.html) | ☐ |
| 2 | | **MoCo**: Momentum Contrast for Unsupervised Visual Representation Learning (He et al.) | CVPR 2020 | Introduces the **momentum (EMA) encoder**, which JEPA reuses as its target encoder | [arXiv](https://arxiv.org/abs/1911.05722) · [official](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html) | ☐ |
| 3 | ★ | **BYOL**: Bootstrap Your Own Latent (Grill et al.) | NeurIPS 2020 | No negatives needed. Online network + predictor + EMA target is **the direct ancestor of JEPA's architecture** | [arXiv](https://arxiv.org/abs/2006.07733) · [official](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html) | ☐ |
| 4 | | **SimSiam**: Exploring Simple Siamese Representation Learning (Chen & He) | CVPR 2021 | Shows why stop-gradient prevents **collapse** (all inputs mapping to the same output) | [arXiv](https://arxiv.org/abs/2011.10566) · [official](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html) | ☐ |
| 5 | | **Barlow Twins**: SSL via Redundancy Reduction (Zbontar et al.) | ICML 2021 | Another way to avoid collapse, by decorrelating features | [arXiv](https://arxiv.org/abs/2103.03230) · [official](https://proceedings.mlr.press/v139/zbontar21a.html) | ☐ |
| 6 | | **DINO**: Emerging Properties in Self-Supervised Vision Transformers (Caron et al.) | ICCV 2021 | Self-distillation with ViTs. DINO features are the frozen encoder in DINO-WM (#26) | [arXiv](https://arxiv.org/abs/2104.14294) · [official](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html) | ☐ |
| 7 | ★ | **VICReg**: Variance-Invariance-Covariance Regularization (Bardes, Ponce, LeCun) | ICLR 2022 | An explicit anti-collapse loss from LeCun's group, and a key idea behind JEPA training | [arXiv](https://arxiv.org/abs/2105.04906) · [official](https://openreview.net/forum?id=xm6YD62D1Ub) | ☐ |
| 8 | ★ | **MAE**: Masked Autoencoders Are Scalable Vision Learners (He et al.) | CVPR 2022 | The "predict the pixels" approach. **Understand it in order to understand what JEPA moves away from** | [arXiv](https://arxiv.org/abs/2111.06377) · [official](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) | ☐ |
| 9 | | **data2vec**: General SSL in Speech, Vision and Language (Baevski et al.) | ICML 2022 | Predicts *latent* targets across modalities. The closest pre-JEPA idea | [arXiv](https://arxiv.org/abs/2202.03555) · [official](https://proceedings.mlr.press/v162/baevski22a.html) | ☐ |

## Stage 2: JEPA

| # | ★ | Paper | Venue | Why read it | Links | Read |
|---|---|-------|-------|-------------|-------|------|
| 10 | ★ | **I-JEPA**: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (Assran et al.) | CVPR 2023 | **The first JEPA.** It predicts the embeddings of masked image regions instead of their pixels | [arXiv](https://arxiv.org/abs/2301.08243) · [official](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) | ☐ |
| 11 | ★ | **V-JEPA**: Revisiting Feature Prediction for Learning Visual Representations from Video (Bardes et al.) | TMLR 2024 (Featured) | Extends JEPA to video, adding **time**, which is the step toward sensor streams | [arXiv](https://arxiv.org/abs/2404.08471) · [official](https://openreview.net/forum?id=QaCCuDfBk2) | ☐ |
| 12 | | How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks (Littwin et al.) | NeurIPS 2024 | Theory of *why* predicting in latent space ignores noise. Read after #10–11 | [arXiv](https://arxiv.org/abs/2407.03475) · [official](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a600f0f740605205133553cb74a1c107-Abstract-Conference.html) | ☐ |

## Stage 3: World models (learn the dynamics, then plan or act in imagination)

| # | ★ | Paper | Venue | Why read it | Links | Read |
|---|---|-------|-------|-------------|-------|------|
| 13 | ★ | **Recurrent World Models Facilitate Policy Evolution** (Ha & Schmidhuber) | NeurIPS 2018 (oral) | The paper that popularised the term "world models". There is also a friendlier arXiv version, [World Models (1803.10122)](https://arxiv.org/abs/1803.10122) | [arXiv](https://arxiv.org/abs/1809.01999) · [official](https://proceedings.neurips.cc/paper_files/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html) | ☐ |
| 14 | | **PETS**: Deep RL in a Handful of Trials using Probabilistic Dynamics Models (Chua et al.) | NeurIPS 2018 | Learned dynamics + **MPC**, the classic control recipe that matters for industrial work | [arXiv](https://arxiv.org/abs/1805.12114) · [official](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) | ☐ |
| 15 | ★ | **PlaNet**: Learning Latent Dynamics for Planning from Pixels (Hafner et al.) | ICML 2019 | Introduces the **RSSM** latent dynamics model and planning in latent space | [arXiv](https://arxiv.org/abs/1811.04551) · [official](https://proceedings.mlr.press/v97/hafner19a.html) | ☐ |
| 16 | | **MBPO**: When to Trust Your Model (Janner et al.) | NeurIPS 2019 | Covers **compounding error** over long rollouts, open problem #2 in the blog | [arXiv](https://arxiv.org/abs/1906.08253) · [official](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html) | ☐ |
| 17 | ★ | **Dreamer**: Dream to Control, Learning Behaviors by Latent Imagination (Hafner et al.) | ICLR 2020 | Learns a policy entirely inside the world model's imagination | [arXiv](https://arxiv.org/abs/1912.01603) · [official](https://openreview.net/forum?id=S1lOTC4tDS) | ☐ |
| 18 | | **MuZero**: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (Schrittwieser et al.) | Nature 2020 | Plans with a model that **never reconstructs observations**, which is philosophically close to JEPA | [arXiv](https://arxiv.org/abs/1911.08265) · [official](https://www.nature.com/articles/s41586-020-03051-4) | ☐ |
| 19 | | **DreamerV2**: Mastering Atari with Discrete World Models (Hafner et al.) | ICLR 2021 | Discrete latents. Optional, since DreamerV3 supersedes it | [arXiv](https://arxiv.org/abs/2010.02193) · [official](https://openreview.net/forum?id=0oabwyZbOu) | ☐ |
| 20 | ★ | **TD-MPC**: Temporal Difference Learning for Model Predictive Control (Hansen et al.) | ICML 2022 | A **decoder-free** latent model + MPC. Very JEPA-like, and the most relevant design for process control | [arXiv](https://arxiv.org/abs/2203.04955) · [official](https://proceedings.mlr.press/v162/hansen22a.html) | ☐ |
| 21 | | **IRIS**: Transformers are Sample-Efficient World Models (Micheli et al.) | ICLR 2023 (top 5%) | A transformer as the world model | [arXiv](https://arxiv.org/abs/2209.00588) · [official](https://openreview.net/forum?id=vhFu1Acb0xb) | ☐ |
| 22 | | **TD-MPC2**: Scalable, Robust World Models for Continuous Control (Hansen et al.) | ICLR 2024 (spotlight) | TD-MPC scaled up to many tasks with one model | [arXiv](https://arxiv.org/abs/2310.16828) · [official](https://openreview.net/forum?id=Oxh5CstDJU) | ☐ |
| 23 | | **Genie**: Generative Interactive Environments (Bruce et al.) | ICML 2024 (oral) | A **generative** world model learned from video. This is the "other family" from the blog | [arXiv](https://arxiv.org/abs/2402.15391) · [official](https://proceedings.mlr.press/v235/bruce24a.html) | ☐ |
| 24 | | **DIAMOND**: Diffusion for World Modeling, Visual Details Matter in Atari (Alonso et al.) | NeurIPS 2024 (spotlight) | A diffusion-based generative world model | [arXiv](https://arxiv.org/abs/2405.12399) · [official](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html) | ☐ |
| 25 | ★ | **DreamerV3**: Mastering diverse control tasks through world models (Hafner et al.) | Nature 2025 | One robust set of hyperparameters across 150+ tasks. The arXiv title is "Mastering Diverse Domains through World Models" | [arXiv](https://arxiv.org/abs/2301.04104) · [official](https://www.nature.com/articles/s41586-025-08744-2) | ☐ |

## Stage 4: Latent world models + planning (where JEPA meets world models)

| # | ★ | Paper | Venue | Why read it | Links | Read |
|---|---|-------|-------|-------------|-------|------|
| 26 | ★ | **DINO-WM**: World Models on Pre-trained Visual Features enable Zero-shot Planning (Zhou et al.) | ICML 2025 | **Frozen pretrained encoder + latent predictor + planning.** The peer-reviewed version of the recipe the blog describes | [arXiv](https://arxiv.org/abs/2411.04983) · [official](https://proceedings.mlr.press/v267/zhou25t.html) | ☐ |
| 27 | | **NWM**: Navigation World Models (Bar et al.) | CVPR 2025 | An action-conditioned world model for robot navigation | [arXiv](https://arxiv.org/abs/2412.03572) · [official](https://openaccess.thecvf.com/content/CVPR2025/html/Bar_Navigation_World_Models_CVPR_2025_paper.html) | ☐ |
| 28 | ★ | **PLDM**: Learning from Reward-Free Offline Data, A Case for Planning with Latent Dynamics Models (Sobal et al.) | NeurIPS 2025 | JEPA-style latent dynamics learned from **offline logs without rewards**, which is exactly what industrial historian data looks like | [arXiv](https://arxiv.org/abs/2502.14819) · [official](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e7cf447f21cd11c846463affefce665-Abstract-Conference.html) | ☐ |

## Stage 5: Self-supervised learning on time series (your data type)

| # | ★ | Paper | Venue | Why read it | Links | Read |
|---|---|-------|-------|-------------|-------|------|
| 29 | | **TST**: A Transformer-based Framework for Multivariate Time Series Representation Learning (Zerveas et al.) | KDD 2021 | Masked pretraining for multivariate sensor data | [arXiv](https://arxiv.org/abs/2010.02803) · [official](https://doi.org/10.1145/3447548.3467401) | ☐ |
| 30 | | **TS-TCC**: Time-Series Representation Learning via Temporal and Contextual Contrasting (Eldele et al.) | IJCAI 2021 | Contrastive SSL adapted to time series | [arXiv](https://arxiv.org/abs/2106.14112) · [official](https://www.ijcai.org/proceedings/2021/324) | ☐ |
| 31 | ★ | **TS2Vec**: Towards Universal Representation of Time Series (Yue et al.) | AAAI 2022 | A standard baseline for time-series representations | [arXiv](https://arxiv.org/abs/2106.10466) · [official](https://ojs.aaai.org/index.php/AAAI/article/view/20881) | ☐ |
| 32 | | **TF-C**: Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency (Zhang et al.) | NeurIPS 2022 | **Transfer across sensor domains**, relevant to the "what transfers?" question | [arXiv](https://arxiv.org/abs/2206.08496) · [official](https://proceedings.neurips.cc/paper_files/paper/2022/hash/194b8dac525581c346e30a2cebe9a369-Abstract-Conference.html) | ☐ |
| 33 | ★ | **PatchTST**: A Time Series is Worth 64 Words (Nie et al.) | ICLR 2023 | **Patching** plus channel independence, now the default way to feed time series to transformers | [arXiv](https://arxiv.org/abs/2211.14730) · [official](https://openreview.net/forum?id=Jbdc0vTOcol) | ☐ |
| 34 | | **SimMTM**: A Simple Pre-Training Framework for Masked Time-Series Modeling (Dong et al.) | NeurIPS 2023 (spotlight) | Masked modelling done right for time series | [arXiv](https://arxiv.org/abs/2302.00861) · [official](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html) | ☐ |
| 35 | ★ | **MOMENT**: A Family of Open Time-series Foundation Models (Goswami et al.) | ICML 2024 | An open time-series foundation model, and a candidate **frozen encoder** for direction D2 | [arXiv](https://arxiv.org/abs/2402.03885) · [official](https://proceedings.mlr.press/v235/goswami24a.html) | ☐ |
| 36 | | **TimesFM**: A decoder-only foundation model for time-series forecasting (Das et al.) | ICML 2024 | Google's time-series foundation model | [arXiv](https://arxiv.org/abs/2310.10688) · [official](https://proceedings.mlr.press/v235/das24c.html) | ☐ |
| 37 | | **Chronos**: Learning the Language of Time Series (Ansari et al.) | TMLR 2024 | Amazon's model, which tokenises time series the way LLMs tokenise text | [arXiv](https://arxiv.org/abs/2403.07815) · [official](https://openreview.net/forum?id=gerNCVqqtR) | ☐ |

## Stage 6: JEPA × time series and industrial control (closest to our topic)

Venue tiers vary in this stage. Read these for their **content**, not their prestige.

| # | ★ | Paper | Venue | Why read it | Links | Read |
|---|---|-------|-------|-------------|-------|------|
| 38 | ★ | **Time-Series JEPA for Predictive Remote Control under Capacity-Limited Networks** (Girgis, Valcarce, Bennis) | IEEE IoT Journal 2026 | A JEPA on time series used **for control** | [arXiv](https://arxiv.org/abs/2406.04853) · [official](https://doi.org/10.1109/JIOT.2025.3650435) | ☐ |
| 39 | ★ | **Learning Actionable World Models for Industrial Process Control** (Yan et al.) | IEEE SDS 2025 | **The closest prior work** to our direction, on injection moulding | [arXiv](https://arxiv.org/abs/2503.01411) · [official](https://doi.org/10.1109/SDS66131.2025.00022) | ☐ |
| 40 | | **SC-JEPA**: Stabilizing Latent Predictive Learning for Time-Series Anomaly Prediction (He et al.) | SDM 2026 (to appear, Nov 2026) | How to stop a JEPA collapsing on time series | [arXiv](https://arxiv.org/abs/2602.04643) · [program](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=91498) | ☐ |
| 41 | | **Joint Embeddings Go Temporal** (Ennadir, Golkar, Sarra) | NeurIPS 2024 Workshop (TSALM) | The "TS-JEPA" often cited by other papers. It is a different paper from #38 despite the same nickname | [arXiv](https://arxiv.org/abs/2509.25449) · [OpenReview](https://openreview.net/forum?id=FIdbozebmy) | ☐ |

---

## Not peer-reviewed, but essential

These are not conference or journal papers. They are worth reading anyway because the field builds on them.

| # | Paper | Status | Why read it | Link | Read |
|---|-------|--------|-------------|------|------|
| E1 | **A Path Towards Autonomous Machine Intelligence** (LeCun, 2022) | Position paper | **Where JEPA comes from.** It is long, so read the sections on world models, JEPA and hierarchical planning | [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | ☐ |
| E2 | **A Cookbook of Self-Supervised Learning** (Balestriero et al., 2023) | arXiv survey | Best beginner overview of the whole SSL field. Read alongside Stage 1 | [arXiv](https://arxiv.org/abs/2304.12210) | ☐ |
| E3 | **CPC**: Representation Learning with Contrastive Predictive Coding (van den Oord et al., 2018) | arXiv only | Predicts the **future in latent space for sequences**, including audio. A direct conceptual ancestor for sensor-stream JEPAs | [arXiv](https://arxiv.org/abs/1807.03748) | ☐ |
| E4 | **V-JEPA 2** (Assran et al., 2025) | arXiv only (as of 2026-10) | Introduces **V-JEPA 2-AC**, the recipe the blog builds on | [arXiv](https://arxiv.org/abs/2506.09985) | ☐ |
| E5 | **RoboJEPA** (Zholus et al., 2026) | arXiv only (Oct 2026) | Scaling laws for latent world models, and imagination error as a proxy | [arXiv](https://arxiv.org/abs/2610.10515) | ☐ |

---

## Suggested order (★ papers only)

| Phase | Topic | Papers |
|-------|-------|--------|
| **1** | Self-supervised learning | E2 (Cookbook, skim) → 1 SimCLR → 3 BYOL → 7 VICReg → 8 MAE |
| **2** | JEPA | E1 (LeCun, JEPA sections) → 10 I-JEPA → 11 V-JEPA → E4 V-JEPA 2 (the AC sections) |
| **3** | World models | 13 Ha & Schmidhuber → 15 PlaNet → 17 Dreamer → 20 TD-MPC → 25 DreamerV3 |
| **4** | JEPA + planning | 26 DINO-WM → 28 PLDM → E5 RoboJEPA |
| **5** | Your domain | 31 TS2Vec → 33 PatchTST → 35 MOMENT → 38 TS-JEPA (control) → 39 Yan et al. |

### How to read each paper
1. **First pass (~15 min):** title, abstract, figures, conclusion. Ask: what problem, what idea, what result?
2. **Second pass (~1 h):** the method section, plus the main results table.
3. **Write 3–5 lines** in `docs/literature/` and tick the ☐ box here.
