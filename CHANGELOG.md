# Changelog

All notable changes to this research notebook are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use
`0.MINOR.PATCH` until a real codebase exists:

- **MINOR**: a milestone, such as a new survey area, a chosen direction or a first experiment.
- **PATCH**: corrections and small additions.

## [Unreleased]

## [0.2.0] - 2026-10-13

### Added
- `docs/reading-list.md`: staged reading list. It covers self-supervised learning, then JEPA, world models,
  latent planning, time-series SSL, and finally JEPA on time series and industrial data.
  - 41 peer-reviewed papers and 5 essential non-peer-reviewed works.
  - Every venue was checked against primary sources (arXiv, PMLR, NeurIPS, CVF, OpenReview, Nature, IEEE, AAAI, IJCAI).
  - Includes a ★ core path and per-paper read checkboxes.

### Changed
- `docs/literature/jepa-sensor-streams.md`: corrected venues.
  - Girgis et al. TS-JEPA is in IEEE IoT Journal 2026, not arXiv-only.
  - SC-JEPA is SDM 2026, to appear.
  - Yan et al. is IEEE SDS 2025.
  - The Ennadir "TS-JEPA" lead is resolved to *Joint Embeddings Go Temporal* (arXiv 2509.25449, NeurIPS 2024 workshop).
- `README.md`: indexed the reading list.

## [0.1.0] - 2026-10-13

### Added
- `README.md`: project overview, doc index, working conventions, remote-push instructions.
- `docs/PROGRESS.md`: progress log with the first session entry.
- `docs/sources/01-blog-llms-world-models.md`: summary of Shamane Siriwardhana's post
  "LLMs, world models, and what comes next" (5 Oct 2026, updated 9 Oct 2026).
- `docs/sources/02-osiris-repo.md`: summary of `simplifaisoul/osiris`. Finding: it is an
  OSINT dashboard, not a JEPA or world-model project.
- `docs/literature/jepa-sensor-streams.md`: related-work table with 10 papers
  (9 at `abstract` level, 1 at `search` level) and 5 unchecked leads.
- `docs/research/thesis-and-directions.md`: assessment of the "world models vs LLMs" claim,
  gap analysis, and 4 candidate contribution directions.
- `.gitignore` for future Python/ML work.
- Local git repository initialised. Tagged `v0.1.0`.
