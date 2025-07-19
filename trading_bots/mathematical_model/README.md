# SKA-Based Probability Framework for Trading Bots

## 1. Introduction

This document presents a theoretical framework for modeling market-regime transitions using entropy-based probabilities (SKA) in a quantum-inspired notation. By employing Dirac (bra–ket) notation, we define market states in a concise and mathematically rigorous way. This README is intended for algorithmic trading developers who wish to integrate regime-switching probabilities into their trading bots.

## 2. Dirac Notation for Market Regimes

In our framework, we represent each market regime as a basis vector (ket) in an abstract state space:

* **Bull Market**: $\lvert\text{bull}\rangle$
* **Neutral Market**: $\lvert\text{neutral}\rangle$
* **Bear Market**: $\lvert\text{bear}\rangle$

These kets satisfy the orthonormality condition:

$$
\Large \langle i \vert j \rangle = \delta_{ij},
$$

where $i,j\in\{\text{bull},\text{neutral},\text{bear}\}$ and $\delta_{ij}$ is the Kronecker delta.

### 2.1 Completeness Relation

The basis spans the full regime space, so we have:

$$
\Large \mathbf{I} = \sum_{r\in\{\text{bull, neutral, bear}\}}\lvert r\rangle\langle r\rvert
$$

where $\mathbf{I}$ is the identity operator.

## 3. Time‐Domain Probability Normalization

Let $\Large T$ be the total length of our observation window. Define an operator $\Large \Delta\tau$ whose matrix elements

$$
\Large \langle i \mid \Delta\tau \mid j\rangle = \Delta\tau_{ij}
$$

represent the total time spent transitioning from regime $\lvert i\rangle$ to $\lvert j\rangle$ during $T$. Then the **total statistical probability** satisfies

$$
\Large \sum_{i,j}\frac{\langle i \mid \Delta\tau \mid j\rangle}{T} = 1.
$$

We may therefore identify the empirical transition probability as

$$
\Large P_{i\to j} = \frac{\langle i \mid \Delta\tau \mid j\rangle}{T},
$$

which guarantees $\Large \sum_{i,j}P_{i\to j}=1$ by construction.

## 4. Trend Regime Pairs

We classify certain back‑and‑forth transitions between regimes as "trends."

| Trend       | Paired Regime Transitions                                      | Variable Name        | Description                                        |
|-------------|----------------------------------------------------------------|----------------------|----------------------------------------------------|
| **Uptrend** | ⟨neutral\|Δτ\|bull⟩ and ⟨bull\|Δτ\|neutral⟩                          | `trend_up_pairs`     | Transitions between Neutral ⇆ Bull                  |
| **Downtrend** | ⟨neutral\|Δτ\|bear⟩ and ⟨bear\|Δτ\|neutral⟩                        | `trend_down_pairs`   | Transitions between Neutral ⇆ Bear                  |

**Notes for developers:**
- Represent regimes by indices or enums (e.g. `0 = neutral`, `1 = bull`, `2 = bear`)
- Store each pair as a 2‑tuple of indices, e.g.:
  - Use `trend_up_pairs = [(0,1), (1,0)]`
  - Use `trend_down_pairs = [(0,2), (2,0)]`
- Checking `(i,j) in trend_up_pairs` identifies uptrends; similarly for downtrends

## 5. Definition of $\{\Delta\tau_{i\to j}(t)\}$

$\{\Delta\tau_{i\to j}(t)\}$ is the **time‑series** of all observed durations for transitions from regime $i$ to regime $j$. Concretely:

1. **Regime sequence**
   You have a sequence of timestamped regime observations

   $$\Large \{(t_1, r_1), (t_2, r_2), \dots, (t_M, r_M)\},\quad r_k\in\{0,1,2\}.$$

2. **Instantaneous transition time**
   Whenever you see a transition $r_k = i$ followed by $r_{k+1} = j$, record

   $$\Large \Delta\tau_{i\to j}\bigl(t_{k+1}\bigr) = t_{k+1} - t_{k},$$

   the elapsed clock‐time between those two events.

3. **Building the series**
   Collect these durations in chronological order:

   $$\Large \{\Delta\tau_{i\to j}(t)\} = \bigl\{\Delta\tau_{i\to j}(t_{k_1}),\; \Delta\tau_{i\to j}(t_{k_2}),\ \dots \bigr\},$$

   where $k_1, k_2, \dots$ index just the steps with an $i\to j$ transition.

**Interpretation:**
* $\Delta\tau_{i\to j}(t)$ is "the time it took, measured at moment $t$, to go from regime $i$ to $j$"
* $\{\Delta\tau_{i\to j}(t)\}$ is the full list of those durations over your data window

## 6. Trend‑Pair Correlation Coefficients (Time‑Windowed)

Introduce a fixed **time window** $T_{\rm corr}$ over which we compute each trend's Pearson correlation. Only transitions occurring within the most recent $T_{\rm corr}$ seconds are included.

### 6.1 Window Parameter

| Math Symbol    | Variable Name          | Description                                            | Unit        |
| -------------- | ---------------------- | ------------------------------------------------------ | ----------- |
| $T_{\rm corr}$ | **corr\_time\_window** | Length of the sliding time window used for correlation | seconds (s) |

### 6.2 Defined Coefficients

Within each window of length $T_{\rm corr}$, collect
$\{\Delta\tau_{i\to j}(t)\}$ and $\{\Delta\tau_{j\to i}(t)\}$ for all $t \in [\,t_{\rm now}-T_{\rm corr},\,t_{\rm now}\,]$. Then define

$$
\rho_{i\!\leftrightarrow\!j} = \mathrm{PearsonCorr}\bigl(\{\Delta\tau_{i\to j}(t)\},\,\{\Delta\tau_{j\to i}(t)\}\bigr).
$$

| Symbol            | Variable Name  | Regime Pairs      | Description                                                          |
| ----------------- | -------------- | ----------------- | -------------------------------------------------------------------- |
| $\rho_{\rm up}$   | **corr\_up**   | (0 → 1) & (1 → 0) | Correlation of Neutral (0)⇆Bull (1) durations in last $T_{\rm corr}$ |
| $\rho_{\rm down}$ | **corr\_down** | (0 → 2) & (2 → 0) | Correlation of Neutral (0)⇆Bear (2) durations in last $T_{\rm corr}$ |

### 6.3 Developer Notes

* **Define your window**
  ```python
  corr_time_window = 3600  # e.g. 1 hour (3600 seconds)
  ```

* **Filter events**
  Only include transitions with timestamp $t$ satisfying
  $$
    t_{\rm now} - t \;\le\; T_{\rm corr}.
  $$

* **Compute correlation**
  Use your preferred statistics library over the two filtered series of `delta_tau[i][j]`.

* **Interpretation**
  * **+1** → perfect synchronization in this window
  * **–1** → perfect anti‑synchronization
  * **0** → no linear relationship