# World Models — Research Notebook

Working notes for exploring **world models** and, specifically, **JEPA-style latent
world models on sensor streams** as a possible research contribution.

> Status: **pre-repo / exploration phase** · Docs version: see [CHANGELOG.md](CHANGELOG.md)
> Started: 2026-10-13

## Origin

A friend suggested that within ~5 years, world models could have more industrial
impact than LLMs have today, and that **JEPA on sensor streams** is a good area to
contribute to. They shared two links, which are summarised under `docs/sources/`.

## Where things are

| Path | What it holds |
|------|---------------|
| [docs/PROGRESS.md](docs/PROGRESS.md) | Dated log: what was done, decisions, next steps. **Start here.** |
| [docs/reading-list.md](docs/reading-list.md) | 41 verified peer-reviewed papers + 5 essential preprints, in 6 stages, with a ★ core path |
| [docs/sources/01-blog-llms-world-models.md](docs/sources/01-blog-llms-world-models.md) | Summary and notes on Shamane Siriwardhana's blog post |
| [docs/sources/02-osiris-repo.md](docs/sources/02-osiris-repo.md) | Summary and assessment of the OSIRIS GitHub repo |
| [docs/literature/jepa-sensor-streams.md](docs/literature/jepa-sensor-streams.md) | Related-work table: JEPA and latent world models on time series and sensors |
| [docs/research/thesis-and-directions.md](docs/research/thesis-and-directions.md) | The "world models vs LLMs" claim, gaps found, and candidate contributions |
| [CHANGELOG.md](CHANGELOG.md) | Versioned history of these docs |

## Working conventions

1. **Every work session** adds a dated entry to `docs/PROGRESS.md`.
2. **Every meaningful doc change** adds a line under `## [Unreleased]` in `CHANGELOG.md`.
   When a milestone is reached, the changes are released as a new version (`0.x.0`) and the commit is tagged.
3. **Commit** after each session with a short, descriptive message.
4. **Mark verification level** on claims about papers:
   - `search` means only a search-result title or snippet was seen.
   - `abstract` means only the arXiv abstract or page was read.
   - `full` means the full paper was read.
   - `code` means the code was run.
5. **Never record paper claims from memory.** Every paper entry must have a link.

## Moving to a remote repo later

This folder is already a local git repository. Once the GitHub repo exists:

```bash
git remote add origin <repo-url>
git push -u origin main --tags
```
