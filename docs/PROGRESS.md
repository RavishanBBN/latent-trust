# Progress Log

Newest entries first. Each entry has these parts: **Done**, **Findings**, **Decisions** and **Next**.

---

## 2026-11-22: Session 4: Reproduction, tests, novelty check, foundations course

**Done**
- Moved to a new machine (copies extracted from the zips) and set up `.venv` (numpy 2.5, scipy 1.18, matplotlib 3.11, pytest; no torch and no GPU yet).
- Reproduced `experiments/run_assignment.py` (504 s). `results.json` matches the 2026-11-14 run except for timing and the last floating-point digits.
- Added `tests/test_filters.py`. All 15 tests (with the gradcheck) pass.
- Committed the 2026-11-14 work and tagged `v0.3.0`.
- Novelty check of K-PWM: abstract read, plus the full HTML text with the middle section read only via a summary.
- Started a foundations course in `learning/` and wrote Lesson 1.

**Findings**
- **K-PWM** (arXiv 2611.11521) differs from D5 on every axis that matters:
  - it uses an MLP decoder, so its measurements are raw sensors, not embeddings;
  - its process and measurement noise covariances are constant and fitted in closed form, not learned per state and input;
  - it has no NIS/NEES calibration and no innovation monitor;
  - it is evaluated on MuJoCo only.
  
  It does already cover a latent EKF + uncertainty-aware MPPI, so D5 must not claim "belief-space MPC" as novel by itself.
- **Lahoud** (SSRN 6521219): SSRN returns 403 to automated fetches, so it is still unread.

**Decisions**
- Learn the foundations first. A ~12-day course at 6–7 h/day (probability → filtering → NN/SSL/JEPA → world models) runs before the research work, so the author can defend every part. The research restarts around 4 Dec 2026, and the preprint target stays at ~9 Jan 2027.

**Next**
- [ ] Lessons 1–4 (`learning/README.md`).
- [ ] Open Lahoud SSRN 6521219 in a browser and record what it does.
- [ ] Read the middle section of the K-PWM PDF before writing the related-work section.
- [ ] After the course: PyTorch port, then RQ1–RQ2 on the quadruple tank.

---

## 2026-11-14: Session 3: Direction chosen (D5), seed experiment, Q1 plan

**Done**
- Ran the EN4594 mini project as the seed experiment (`assignment/report.pdf`, code in `src/lbf`).
  - Physics EKF on the quadruple tank.
  - NumPy JEPA world model, plus an EKF in its latent space.
- Verified ~45 new references and the dataset facts (two literature passes, checked against primary pages).
- Wrote the D5 Q1 plan, a journal fee table and `CLAUDE.md` for the move to Claude Code.

**Findings**
- Physics EKF results:
  - Consistent: ANEES 4.10, ANIS 2.01, inside their χ² bands 95 % of the time.
  - RMSE 0.038 cm.
  - The NIS monitor detects a hidden-tank leak in 36 s and a sensor bias in 5 s, with no false alarms.
- Latent EKF results:
  - Recovers the unmeasured level h₃ to 0.107 cm without physics.
  - tr R_lat (0.072) > tr Q_lat (0.023): the learned predictor is less accurate than the embedding noise, so the filter gains only ~2 %.
  - The latent NIS spikes at input steps, which hurts detection: a 0.02 cm² leak is detected in 23 % of runs, rising to 87 % at 0.08 cm².
- **The bottleneck is calibrating the world model's uncertainty.** That becomes RQ1–RQ2.
- Closest prior work: **K-PWM** (NeurIPS 2026, arXiv 2611.11521) runs a latent EKF with MPPI, but is decoder-based and has no monitor. MTS3, KVAE, RKN, Deshpande (L-CSS 2023) and Falconer (Physica D 2025) are also close.
- **PRONTO** (real rig, CC BY 4.0) is the best free real dataset that logs control inputs and has labelled faults.

**Decisions**
- D5 chosen as the primary research direction. Target journal: J. Process Control or EAAI (free); stretch: IEEE TNNLS.
- Earthquake-early-warning particle-filter idea kept as a second project (separate repo `eew-particle-filter`).

**Next**
- [ ] Commit this session (see CLAUDE.md rule 6), tag v0.3.0.
- [ ] Port to PyTorch; add the noise head R(s,u) and β-NLL; handle correlated measurement noise; UKF; adaptive Q/R.
- [ ] Build the TEP pipeline and baselines; post an arXiv preprint by about 9 Jan 2027.
- [ ] Read K-PWM and the SSRN Lahoud paper in full before writing any novelty claim.

---

## 2026-10-13: Session 2: Reading list for fundamentals

**Done**
- Built `docs/reading-list.md`, a staged path from self-supervised learning to JEPA, world models,
  latent planning, time-series SSL and finally JEPA on time series.
  - 41 peer-reviewed papers + 5 essential preprints, with a ★ core path of ~22 papers.
  - Venues were verified against primary sources by three parallel checks covering ~50 papers.
- Corrected venues in the literature table.

**Findings**
- Several important works are **not peer-reviewed** as of Oct 2026. They are kept in a separate "essential" table:
  - V-JEPA 2 (arXiv only)
  - LeCun's JEPA position paper
  - RoboJEPA
  - CPC
  - LeJEPA
  - Cosmos (tech report)
- **DINO-WM (ICML 2025)** and **PLDM (NeurIPS 2025)** are peer-reviewed versions of the "frozen encoder +
  latent predictor + planning" recipe. Cite these alongside V-JEPA 2-AC.
- "TS-JEPA" is the nickname of two different papers:
  - Girgis et al., IEEE IoT J 2026
  - Ennadir et al., NeurIPS 2024 workshop

  Always cite them by author.
- Some arXiv "Comments" fields give the wrong venue. For example, I-JEPA's says ICCV, but the paper is CVPR 2023. Always cite the proceedings.

**Next**
- [ ] Start Phase 1 of the reading list (SSL basics).
- [ ] The open items from Session 1 still stand.

---

## 2026-10-13: Session 1: Kickoff and source review

**Done**
- Read both links the friend shared:
  - Blog: Shamane Siriwardhana, *LLMs, world models, and what comes next* → `docs/sources/01-blog-llms-world-models.md`
  - Repo: `simplifaisoul/osiris` → `docs/sources/02-osiris-repo.md`
- Surveyed JEPA and latent world-model work on sensors and time series. Logged 10 papers (9 read at abstract level) and 5 unchecked leads
  → `docs/literature/jepa-sensor-streams.md`
- Wrote an assessment of the "world models vs LLMs" claim and 4 candidate directions → `docs/research/thesis-and-directions.md`
- Set up the docs structure, CHANGELOG and a local git repo (tag `v0.1.0`).

**Findings**
- **OSIRIS is not a world-model or JEPA project.** It is a real-time OSINT dashboard (Next.js and MapLibre). Its
  "parallel simulated worlds" feature means LLM-agent role-play, not learned dynamics. At most it is relevant as an example of,
  or a gateway to, public sensor feeds.
- **The blog doesn't claim that world models will beat LLMs.** It argues that the two are composed in one system, and names
  **industrial sensor streams with control inputs** as a domain-adaptation target.
- **Passive JEPA on time series is already crowded in 2026** (SC-JEPA, CF-JEPA, Phys-JEPA, H-JEPA).
- **Action-conditioned JEPA with planning on industrial streams is mostly open.** The closest work is Yan et al. 2025
  (injection moulding, one process, no public code or data).

**Decisions**
- Docs are kept as Markdown in this folder, versioned with git, and will be pushed when the remote repo exists.
- Claims are tagged with a verification level (`search` / `abstract` / `full` / `code`).

**Next**
- [ ] Ask the friend why they shared OSIRIS and what "sensor streams" means to them.
- [ ] Read RoboJEPA and Yan et al. 2025 in full.
- [ ] Verify the 5 leads in section D of the literature doc.
- [ ] Find public industrial datasets or simulators that **log control inputs** (Tennessee Eastman, SWaT/WADI, HVAC).
- [ ] Settle the compute budget, which decides between directions D1/D3 and D2.
