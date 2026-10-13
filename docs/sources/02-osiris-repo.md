# Source 02: OSIRIS (`simplifaisoul/osiris`)

| Field | Value |
|-------|-------|
| URL | https://github.com/simplifaisoul/osiris |
| Live demo | https://osirisai.live |
| License | MIT |
| Popularity (at access) | ~10.9k stars, ~2.3k forks, 337 commits |
| Accessed | 2026-10-13 |
| Shared by | friend, alongside source 01 |

## What it is

OSIRIS describes itself as an *"Open Source Global Intelligence Platform: Real-Time OSINT Dashboard,
a Palantir Alternative."* It aggregates live public feeds into a single GPU-accelerated map.

**It is not a JEPA model, a world model or an ML training project.** It is a web application.

## Stack
- Next.js 16 (App Router, Turbopack), TypeScript 5
- MapLibre GL JS (WebGL, 60 fps with thousands of entities)
- Framer Motion, Lucide React
- Deployment: Vercel, or Docker (Node 22-alpine)
- An MCP server at `/api/mcp` lets agents such as Claude and Cursor query it

## Data layers (16 toggleable)
| Domain | Source |
|--------|--------|
| Aviation (commercial, private, military) | OpenSky Network |
| Seismic (M2.5+) | USGS |
| Active fires | NASA FIRMS |
| Weather | NOAA |
| CCTV (17k+ cameras) | TfL, WSDOT, Caltrans, TxDOT, regional DOTs |
| Maritime | 39 ports (static intel) |
| News | 23 broadcasters |
| Crypto tracing | blockstream.info, Blockscout |
| Sanctions | OpenSanctions (OFAC SDN) |
| Telegram | public channel previews |

The README gives no per-feed polling intervals. It mentions only a general relaxation to 15–30 min polling.

## The "OI" engine, a different meaning of "world model"
- **OI Assist**: a natural-language interface to the map, using the user's own LLM key.
- **OI Forecast**: *"cast the actors who shape an outcome and play them against each other in
  parallel simulated worlds"*. It weights the results against prediction markets and baselines, and
  is inspired by [Pythia](https://github.com/jangles-byte/Pythia) and
  [MiroFish](https://github.com/666ghj/MiroFish) (swarm-intelligence prediction).

⚠️ **Terminology clash.** Here "simulated worlds" means **LLM-agent role-play simulations**.
These are not learned dynamics models like JEPA or Dreamer. Keep the two meanings separate in any
writing.

## Why might the friend have shared it? (my guesses, to confirm with them)
1. **As an example of real-time multi-sensor streams.** It shows heterogeneous, irregular, multi-rate,
   geo-tagged streams (flights, quakes, fires, weather), the kind of messy input a
   sensor-stream world model would face.
2. **As a source of public data feeds.** Its upstream APIs (USGS, NASA FIRMS, OpenSky, NOAA) are public
   and could be used directly, without going through OSIRIS.
3. **As an application layer.** A world model's predictions could be surfaced in a dashboard like this.

## Relevance assessment
| Aspect | Relevance |
|--------|-----------|
| Method (JEPA / world models) | **None** |
| Data (sensor streams) | **Medium.** The upstream feeds are useful, but most are *passive* (no control inputs or actions), so they suit forecasting and representation learning, not action-conditioned world models. |
| Engineering (ingesting live streams) | Medium. Useful as a reference for building a data-collection pipeline. |

## Cautions if it is used for data
- Respect each upstream source's terms of use and rate limits. Some sources (OpenSky, FIRMS) need keys for higher volume.
- CCTV and Telegram layers raise privacy concerns. Avoid them for research datasets.
- The repo also ships a "RECON toolkit" (port scanner, WHOIS and similar tools). This is unrelated to our work.

## Action item
- [ ] Ask the friend why they shared OSIRIS: as a data source, as an application idea, or something else?
