# Source 01: "LLMs, world models, and what comes next"

| Field | Value |
|-------|-------|
| URL | https://shamanez.github.io/blog/general-brains-domain-adapted-worlds.html |
| Author | Shamane Siriwardhana (Shamane's Corner) |
| Published | 2026-10-05 (updated 2026-10-09) |
| Accessed | 2026-10-13 |
| Shared by | friend, as motivation for "JEPA on sensor streams" |

## Thesis in one line

LLMs keep improving, and the next frontier is connecting that general intelligence to
physical environments through world models. The author's own research focus is the
**domain-adaptation gap**: how to reuse what general models learned while adapting predictions
to a specific domain.

> Note: the post does **not** claim that world models will surpass LLMs in impact, and it gives
> no timeline. It frames LLMs and world models as **complementary parts of one system**.
> The "more impact than LLMs in 5 years" claim comes from the friend, not from this post.

## Structure of the argument

### 1. Frontier foundation models
- The field progressed from self-supervised learning to foundation models, then LLMs, then omni models (text, vision and audio).
- Long-horizon agents such as Claude Code, Codex and computer-use agents are advancing, helped by RL and on-policy distillation.
- Feedback loop: "More useful work produces more outcomes to evaluate. Good examples and
  failures can become training material."
- METR time-horizon measurements show growing task horizons on software tasks.

### 2. Two families of world models
| Family | What it predicts | Examples cited |
|--------|------------------|----------------|
| **Generative / spatial** | Scenes, 3D structure, future observations (pixels) | NVIDIA Cosmos 3 (incl. 4B Edge), Odyssey-3 |
| **Latent predictive (JEPA-style)** | Learned representations, not pixels | V-JEPA 2-AC, RoboJEPA (8B) |

RoboJEPA showed that latent prediction and video rendering can coexist, because predicted latents can be
decoded to video. The post also notes that planning still takes "seconds on multiple GPUs".

### 3. System composition
The proposed system works in four steps:

1. The **LLM** turns a request into goals and coordinates the work.
2. **Generative world models** provide scenes and scenarios.
3. **Latent predictive models** estimate the effects of actions.
4. A **planner** chooses behaviour, and a **controller** executes it.

The post adds that one model may cover several roles, and that not every action needs both kinds of world model at runtime.

### 4. Domain adaptation, the stated research gap
Three papers show complementary routes:
- **V-JEPA 2-AC**: freeze a broad video encoder and learn a robot-specific action-conditioned predictor.
- **JEPA-Anything**: one shared predictive design tested across 7 domains.
- **AD-LiST-JEPA**: latent prediction adapted to LiDAR point clouds (occupancy completion and forecasting).

Core question: *what transfers, and what does a specific domain still need?*

### Domains the post lists for adaptation
- **Robots**: cameras, objects, contacts, motion.
- **LiDAR**: geometry and occupancy, without pixel rendering.
- **Geospatial / multispectral**: place, season, time-scale.
- **Industrial sensor streams**: "Process state and control inputs. Model the dynamics that
  matter for equipment and operating conditions." ← **this is the hook for our direction**

## Open problems named in the post
1. Reusing foundation-model knowledge while adapting to particular domains.
2. Growth of prediction error over longer rollouts.
3. Compatible interfaces between goals, observations and actions.
4. Meeting the latency and reliability needs of deployed systems.
5. Knowing which feedback signals are actually useful, since usage alone doesn't guarantee learning.

## References from the post (verbatim list)
1. Yifeng Yin, *Why Do We Need World Models When LLMs Exist?*, Forbes Tech Council, 2026-10-02. https://www.forbes.com/councils/forbestechcouncil/2026/10/02/why-do-we-need-world-models-when-llms-exist-how-world-models-will-impact-physical-ai/
2. Fei-Fei Li, *From Words to Worlds*, 2025-11-10. https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence
3. World Labs, *A Functional Taxonomy of World Models*, 2026-06-03. https://www.worldlabs.ai/blog/taxonomy-of-world-models
4. METR, *Task-Completion Time Horizons of Frontier AI Models*. https://metr.org/time-horizons/
5. NVIDIA, *Cosmos 3*, 2026. https://research.nvidia.com/labs/cosmos-lab/cosmos3/
6. Cameron & Hawke, *Introducing Odyssey-3*, 2026-09-15. https://odyssey.systems/introducing-odyssey-3
7. Assran et al., *V-JEPA 2*, 2025. https://arxiv.org/abs/2506.09985
8. Zholus et al., *RoboJEPA*, 2026. https://arxiv.org/abs/2610.10515
9. AMI Labs, research direction. https://amilabs.xyz/
10. Cui et al., *JEPA-Anything*, 2026. https://arxiv.org/abs/2609.20800
11. Zhu & Choromanska, *Self-Supervised JEPA-based World Models for LiDAR Occupancy Completion and Forecasting*, 2026. https://arxiv.org/abs/2602.12540
12. DeepSeek-AI, *DeepSeek-R1*, 2025. https://arxiv.org/abs/2501.12948
13. Lu / Thinking Machines Lab, *On-Policy Distillation*, 2025-10-27. https://thinkingmachines.ai/blog/on-policy-distillation/

## My takeaways
- The post supports the "JEPA on sensor streams" idea directly: it names industrial sensor streams with
  **control inputs** as a domain-adaptation target.
- The key words are **adaptation** and **control inputs**. Doing passive forecasting with a JEPA is not
  the gap the post describes. The gap is action-conditioned prediction that reuses general pretraining.
- To read next: references 3 (World Labs taxonomy), 8 (RoboJEPA, released code) and 10 (JEPA-Anything, released code).
