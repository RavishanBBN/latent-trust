# Lesson 1: Gaussians, Bayes' rule and the chi-squared distribution

**Time:** Days 1–2 (about 12 hours, including videos and exercises)

**Goal:** by the end you can explain three things in your own report:
- why the EKF's covariance update has the form $G\Sigma G^\top + R$;
- why the Kalman gain is a "trust ratio";
- why your physics EKF's ANIS came out as **2.01** and its ANEES as **4.10**, and why those are *good* numbers.

**How to use this lesson:**
1. Read a section.
2. Watch or read the matching resource from the list below.
3. Do the exercise for that section in `ex01_gaussians.py`.
4. Ask me about anything unclear, at any point.

---

## Watch / Read (in this order)

| # | Resource | What to take from it | Time |
|---|---|---|---|
| 1 | **3Blue1Brown**, *Bayes theorem, the geometry of changing beliefs* (YouTube) | Intuition for prior → evidence → posterior | 15 min |
| 2 | **StatQuest** videos on the normal distribution, and on covariance and correlation (YouTube) | Mean, variance, standard deviation; what covariance means | 30 min |
| 3 | **Roger Labbe**, *Kalman and Bayesian Filters in Python*: free online Jupyter book (GitHub: `rlabbe/Kalman-and-Bayesian-Filters-in-Python`). Read ch. 2 *Discrete Bayes Filter*, ch. 3 *Probabilities, Gaussians, and Bayes' Theorem*, ch. 5 *Multivariate Gaussians* | The best hands-on introduction; it builds exactly toward the Kalman filter | 4–5 h |
| 4 | **Thrun, Burgard & Fox**, *Probabilistic Robotics*, ch. 2 (*Recursive State Estimation*), especially §2.2 (probability basics) and §2.4 (Bayes filter) | Your course's notation, which our code uses | 2 h |
| 5 | **Khan Academy**, *Chi-square distribution introduction* | What the χ² distribution is | 15 min |
| 6 | Optional, if matrices feel rusty: **3Blue1Brown**, *Essence of Linear Algebra*, episodes 1–4 and 7 | Matrices as transformations, matrix multiplication, inverses | 1 h |

---

## 1. Random variables, mean and variance

A **random variable** is a quantity whose value is uncertain, such as the true water level $h_1$ in tank 1 or a sensor reading $z$.

- The **mean** $\mu = \mathbb{E}[x]$ is its average value.
- The **variance** $\sigma^2 = \mathbb{E}[(x-\mu)^2]$ measures how spread out it is.
- The **standard deviation** $\sigma$ is the square root of the variance, so it has the same units as $x$.

In our project the level sensor has $\sigma = 0.15$ cm, so its variance is $0.0225\ \text{cm}^2$. This is `SIG_V` in `experiments/run_assignment.py`.

### The Gaussian (normal) distribution

$$
p(x) = \mathcal{N}(x;\ \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Two facts about it matter for us:
- About 68 % of the probability lies within $\pm 1\sigma$ of the mean, and about 95 % within $\pm 2\sigma$.
- It is the only distribution that stays a Gaussian under every operation a Kalman filter performs. That is why the Kalman filter needs only two numbers, a mean and a covariance, to describe its whole belief.

---

## 2. Many variables: mean vector and covariance matrix

The quadruple tank has four levels, so the state is a vector $x = [h_1, h_2, h_3, h_4]^\top$. Its uncertainty is described by:
- a **mean vector** $\mu \in \mathbb{R}^4$;
- a **covariance matrix** $\Sigma \in \mathbb{R}^{4\times 4}$, where

$$
\Sigma_{ij} = \mathbb{E}\big[(x_i-\mu_i)(x_j-\mu_j)\big]
$$

How to read $\Sigma$:
- **Diagonal** entries $\Sigma_{ii}$ are the variances of each variable.
- **Off-diagonal** entries $\Sigma_{ij}$ say how two variables move together. If $\Sigma_{13} > 0$, then when $h_1$ is higher than we think, $h_3$ probably is too.
- The **correlation** $\rho_{ij} = \Sigma_{ij} / \sqrt{\Sigma_{ii}\Sigma_{jj}}$ rescales this to the range $-1$ to $1$.

> **Why off-diagonals matter for us.** We only *measure* $h_1$ and $h_2$, yet the EKF still estimates $h_3$ and $h_4$ accurately (RMSE about 0.035 cm). It can do that because the covariance matrix links the unmeasured tanks to the measured ones. When a measurement corrects $h_1$, the off-diagonal terms carry part of that correction over to $h_3$.

The multivariate Gaussian density is

$$
\mathcal{N}(x;\ \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\!\left(-\tfrac12 (x-\mu)^\top \Sigma^{-1} (x-\mu)\right)
$$

Its contours of equal density are **ellipses** (ellipsoids in more than two dimensions). The covariance sets the shape and tilt of each ellipse.

**→ Do Exercise 1.**

---

## 3. The linear-transform rule: this *is* the Kalman predict step

**Rule.** If $x \sim \mathcal{N}(\mu, \Sigma)$ and $y = Ax + b$, then

$$
y \sim \mathcal{N}(A\mu + b,\ A\Sigma A^\top)
$$

**Rule.** If you add independent Gaussian noise $\varepsilon \sim \mathcal{N}(0, R)$, the covariances add:

$$
y = Ax + b + \varepsilon \;\Rightarrow\; y \sim \mathcal{N}(A\mu + b,\ A\Sigma A^\top + R)
$$

Now look at lines 1–2 of the EKF in `src/lbf/ekf.py`:

```
mu_bar    = g(u_t, mu_{t-1})                 # move the mean through the model
Sigma_bar = G_t Sigma_{t-1} G_t^T + R_t      # A Sigma A^T + R, with A = G_t
```

That is exactly these two rules. $G_t$ is the Jacobian of the motion model: the best linear approximation of $g$ near the current estimate. You'll meet Jacobians properly in Lesson 2.

**Intuition:** prediction always *adds* uncertainty, through the $+R$ term. If you stop measuring, the ellipse keeps growing. This is the "error grows with rollout horizon" problem from the world-model blog, written as one equation.

**→ Do Exercise 2.**

---

## 4. Bayes' rule: combining a belief with a measurement

$$
\underbrace{p(x \mid z)}_{\text{posterior}} = \frac{\overbrace{p(z \mid x)}^{\text{likelihood}}\ \overbrace{p(x)}^{\text{prior}}}{p(z)} \;\propto\; p(z \mid x)\, p(x)
$$

The terms mean:
- **Prior** $p(x)$: what you believed before the measurement (the prediction).
- **Likelihood** $p(z \mid x)$: how probable this sensor reading would be if the true state were $x$. This is the sensor model.
- **Posterior** $p(x \mid z)$: the updated belief.
- **Normaliser** $p(z)$: makes the posterior integrate to 1.

### The key result: multiplying two Gaussians gives the Kalman update

Take a 1-D example:
- **Prior:** the tank level is $x \sim \mathcal{N}(\mu_0, \sigma_0^2)$.
- **Sensor:** it reads $z = x + \delta$ with $\delta \sim \mathcal{N}(0, \sigma_z^2)$.

Multiplying the two Gaussians and normalising gives another Gaussian:

$$
K = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_z^2}, \qquad
\mu = \mu_0 + K\,(z - \mu_0), \qquad
\sigma^2 = (1-K)\,\sigma_0^2
$$

That is the **1-D Kalman filter update**. In words:
- **Innovation** $z - \mu_0$: how surprised we are by the measurement.
- **Kalman gain** $K \in [0,1]$: how much we trust the sensor compared with our prediction.
  - Very noisy sensor ($\sigma_z \to \infty$): $K \to 0$, so we ignore it.
  - Very uncertain prior ($\sigma_0 \to \infty$): $K \to 1$, so we jump to the measurement.
- **Posterior variance** is always *smaller* than both $\sigma_0^2$ and $\sigma_z^2$. Combining two sources of information always helps.

**Worked example.** Prior level $12.0$ cm with $\sigma_0 = 1$ cm. The sensor reads $12.6$ cm with $\sigma_z = 0.5$ cm.
- $K = 1/(1 + 0.25) = 0.8$
- $\mu = 12.0 + 0.8 \times 0.6 = 12.48$ cm
- $\sigma^2 = 0.2\ \text{cm}^2$, so $\sigma \approx 0.45$ cm, which is better than either source alone.

The multivariate version, lines 3–5 of `ekf.py`, is the same idea written with matrices:
- $K = \bar\Sigma H^\top (H\bar\Sigma H^\top + Q)^{-1}$
- $\mu = \bar\mu + K(z - h(\bar\mu))$

> **Connection to our research problem.** In the latent EKF, the "prediction" is the JEPA predictor and the "sensor" is the encoder. The predictor's error, $\operatorname{tr} R_{lat} = 0.072$, is *larger* than the encoder's noise, $\operatorname{tr} Q_{lat} = 0.023$. So $K$ is close to 1 and the filter mostly copies the encoder, which is why it gains only about 2 %. Using just this lesson, you can already explain the central weakness in the seed result.

**→ Do Exercise 3.**

---

## 5. The chi-squared distribution and why NIS is a trust test

**Definition.** If $n_1, \dots, n_d$ are independent standard normal variables, $\mathcal{N}(0,1)$, then

$$
n_1^2 + \dots + n_d^2 \sim \chi^2_d \qquad \text{(mean } d\text{, variance } 2d\text{)}
$$

**The fact we use.** If $x \sim \mathcal{N}(\mu, \Sigma)$ in $d$ dimensions, then the squared Mahalanobis distance

$$
m = (x-\mu)^\top \Sigma^{-1} (x-\mu) \;\sim\; \chi^2_d
$$

Multiplying by $\Sigma^{-1}$ "whitens" the error: it rescales and rotates it until it looks like $d$ independent standard normals.

The filter applies this to two quantities:

| Statistic | Formula | What it is | If the filter is right |
|---|---|---|---|
| **NIS** (normalised innovation squared) | $\nu^\top S^{-1} \nu$, with $\nu = z - h(\bar\mu)$ and $S = H\bar\Sigma H^\top + Q$ | How surprising each measurement was, *relative to how surprised the filter expected to be* | $\sim \chi^2_{\dim z}$, mean = number of measurements |
| **NEES** (normalised estimation error squared) | $(x - \mu)^\top \Sigma^{-1} (x - \mu)$ | The true error relative to the claimed uncertainty. Needs the true state, so it only works in simulation | $\sim \chi^2_{\dim x}$, mean = state dimension |

**Your report's numbers:**
- We measure 2 levels, so the average NIS should be about **2**. It was **2.01**.
- The state has 4 levels, so the average NEES should be about **4**. It was **4.10**.

Both mean the filter's claimed uncertainty matches its real errors. The filter is **consistent**, or *calibrated*.

**Using NIS as an alarm.** Under normal operation, NIS exceeds `chi2.ppf(0.99, d)` only 1 % of the time. If NIS stays above that level much more often, *something has changed*: a leak, a sensor fault, or a model that no longer fits. That is the whole idea of our trust monitor.

**What goes wrong in the latent EKF.** Its NIS spikes whenever the pump inputs step. The model's errors are larger than its claimed covariance at exactly those moments, so it raises an alarm when nothing is broken. To avoid false alarms, the threshold has to go up, and then real leaks are missed. Making NIS behave like a true $\chi^2$ through those transients is **RQ2 of the research**.

**→ Do Exercise 4.**

---

## Self-check questions

Try answering these without looking. The answers are at the bottom of `solutions/ex01_gaussians_solution.py`.

1. Why does the predicted covariance always grow, and what single term causes it?
2. A sensor gets twice as noisy ($\sigma_z$ doubles). Does $K$ go up or down? What happens to the posterior variance?
3. We measure only $h_1$ and $h_2$. What, inside $\Sigma$, lets the filter still estimate $h_3$?
4. A filter's average NIS over many runs is 6, with 2 measurements. Is it overconfident or underconfident? Which way is $Q$ or $R$ probably wrong?
5. In one sentence: why does the latent EKF barely improve on the raw encoder?

---

## Exercises

From the repo root, run:

```bash
.venv/bin/python learning/01_probability/ex01_gaussians.py
```

- Fill in every `TODO`. Each exercise prints **PASS** or **FAIL** and saves a figure to `learning/01_probability/figures/`.
- Try for at least 20 minutes before opening `solutions/`.
- When everything passes, tell me. We'll go through the self-check questions together, then start Lesson 2 (the full Kalman filter and EKF on the quadruple tank).
