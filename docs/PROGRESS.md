# Progress Log

Newest entries first. Each entry has these parts: **Done**, **Findings**, **Decisions** and **Next**.

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
