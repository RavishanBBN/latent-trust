# Progress Log

Newest entries first. Each entry has these parts: **Done**, **Findings**, **Decisions** and **Next**.

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
