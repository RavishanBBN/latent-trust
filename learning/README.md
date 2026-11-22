# Foundations course for D5 (started 2026-11-22)

A 12-day path, at 6–7 h/day, from basics to understanding every part of this project.

**How each lesson works:**
- `LESSON.md` holds the explanation, a *Watch / Read* list (videos and book chapters), self-check questions, and links to our own code.
- `exNN_*.py` holds exercises with `TODO`s that print PASS or FAIL.
- `solutions/` holds worked solutions and the self-check answers.

| Days | Lesson | Checkpoint: you can explain… | Status |
|---|---|---|---|
| 1–2 | [01 Probability: Gaussians, Bayes, χ²](01_probability/LESSON.md) | the χ² bands, ANIS 2.01 and ANEES 4.10 in the report | ⏳ |
| 3–6 | 02 State estimation: KF, EKF, NEES/NIS, UKF/PF, CUSUM | all of Part A of the report; `ekf.py`, `quadtank.py`, `metrics.py` | — |
| 7–10 | 03 Neural nets and SSL: backprop, PyTorch, BYOL/EMA, VICReg, JEPA | the training loss in `jepa.py`, term by term | — |
| 11–12 | 04 World models: rollouts, latent EKF, MPC, PCA T²/SPE | Part B of the report, and why the latent filter gains only 2 % | — |

Run the exercises from the repo root: `.venv/bin/python learning/01_probability/ex01_gaussians.py`
