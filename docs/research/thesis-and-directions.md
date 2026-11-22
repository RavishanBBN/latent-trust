# Thesis and Candidate Directions

Last updated: 2026-11-14 · Status: **D5 chosen** (see `D5-latent-bayesian-filtering.md`)

## 1. The claim under consideration

> "In ~5 years, world models will have more industrial impact than LLMs have today."
> (from a friend, 2026-10)

### What supports it
- **Scaling looks predictable for latent world models.** RoboJEPA (2026-10) reports a power law
  between compute and imagination error, and shows that imagination error tracks real-robot planning
  success. Predictable scaling is what made investment in LLMs rational.
- **A small amount of domain data goes far.** V-JEPA 2-AC needed **<62 h** of unlabelled robot video
  on top of a general video encoder.
- **Industry and lab momentum is real.** Examples include NVIDIA Cosmos 3, Odyssey-3, World Labs, AMI
  Labs, a Forbes Tech Council piece (2026-10) and Fei-Fei Li's "spatial intelligence" essay.
- **Physical industries are large and under-served by text models.** Manufacturing, energy, process
  control and logistics are all sensor-driven.

### What cuts against it, or remains unproven
- **The source blog does not make this claim.** It frames world models as one part of a system next to
  LLMs (LLM → goals; world models → consequences; planner and controller → actions), not as a replacement.
- **Action-labelled data is scarce.** Industrial control logs are proprietary, safety-critical and
  heterogeneous across plants.
- **Rollout error grows with the horizon, and latency is still high.** RoboJEPA planning takes "seconds on multiple GPUs".
- **Deployment bars are higher in physical systems.** Certification and safety requirements are strict, and mistakes are costly.
- **The comparison itself is fuzzy.** "Impact" is hard to measure, and LLMs will keep improving over the same 5 years.

### Working position
The weaker version is defensible and is enough to justify the research: *world models will become a
necessary component for AI in physical industries, and domain adaptation is a bottleneck.* Avoid
arguing the strong "more than LLMs" version in any write-up.

## 2. Gap analysis (from `docs/literature/jepa-sensor-streams.md`)

| Ingredient | Vision / robotics | Industrial sensor streams |
|-----------|-------------------|---------------------------|
| General pretrained encoder | ✅ V-JEPA 2 | ? (TS foundation models exist but are untested as JEPA world-model encoders) |
| Action-conditioned predictor | ✅ V-JEPA 2-AC, RoboJEPA | ⚠️ Yan et al. 2025 only (one process, from scratch) |
| Planning with the latent model | ✅ zero-shot on robots | ❌ not found |
| Imagination error as an eval proxy | ✅ RoboJEPA | ❌ untested |
| Public benchmark | ✅ DROID, PushT, RoboTwin | ❌ none for AC latent world models |
| Passive JEPA (forecast / anomaly) | n/a | ✅ crowded (SC-, CF-, Phys-, H-JEPA) |

**Gap:** the V-JEPA 2-AC recipe (general encoder → domain-specific action-conditioned predictor →
planning) has not yet been shown for industrial sensor streams with control inputs.

## 3. Candidate contributions (to discuss with friend and supervisor)

### D1. AC-JEPA for process control on a public benchmark ⭐ current favourite
Train an action-conditioned JEPA world model on a public process-control environment, such as the
Tennessee Eastman Process (*to verify: availability, licence, control-input logging*). Use it for MPC
planning, and compare against a pixel-free generative baseline (e.g., a Dreamer-style RSSM) and
classic system identification.
- *Why it's good:* it is reproducible, unlike Yan et al. (2025), and it fills the "planning" gap.
- *Risk:* simulated processes may be too clean, so it is unclear whether results transfer to real plants.

### D2. "What transfers?" Domain adaptation from a general time-series encoder
Use a frozen, general time-series foundation encoder and a lightweight action-conditioned predictor,
mirroring V-JEPA 2-AC. Measure how much plant-specific data is needed compared with training from scratch.
- *Why it's good:* this is exactly the blog's stated open question.
- *Risk:* existing TS foundation models may not produce latents suited to control.

### D3. Is imagination error a proxy for control quality in process systems?
Replicate RoboJEPA's proxy finding for process control: does latent rollout error predict
closed-loop performance?
- *Why it's good:* the question is small, sharp and falsifiable, and it is cheap compared with scaling studies.
- *Risk:* it may come out as a negative result. That is still publishable as a workshop paper.

### D4. Multi-rate, irregular, heterogeneous sensor streams
Build a JEPA that handles asynchronous sensors at different sampling rates. OSIRIS-style feeds
(USGS, FIRMS, NOAA, OpenSky) would provide passive data for representation pretraining.
- *Why it's good:* real sensor data is messy, and most TS-JEPAs assume regular, aligned channels.
- *Risk:* there are no actions in this data, so it drifts back into the crowded passive-forecasting area.

### D5. Latent Bayesian filtering + innovation-based trust monitoring ⭐⭐ CHOSEN (2026-11-14)
Run an EKF/UKF/PF inside the latent space of a decoder-free action-conditioned JEPA, with encoder embeddings as
measurements, make the innovation statistic (NIS) calibrated, and use it as a run-time trust/fault monitor and as
the belief for MPC. It combines D1 (AC-JEPA on process control), D3 (a principled version of the imagination-error
proxy) and the blog's open problems #2 and #4. Seed results come from the EN4594 mini project. Full plan in
`D5-latent-bayesian-filtering.md`.

## 4. Questions to resolve next
1. What compute is available (GPU type and hours)? This decides between D1/D3 and D2.
2. Is there access to any **real** industrial data with control logs (university labs, industry contacts)?
3. What does the friend mean by "sensor streams": industrial process, IoT, wearables or geospatial?
4. Target venue and timeline (workshop paper, thesis chapter or full conference)?
