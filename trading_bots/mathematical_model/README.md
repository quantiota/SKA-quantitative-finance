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

   $\Large \{(t_1, r_1), (t_2, r_2), \dots, (t_M, r_M)\}, \quad r_k\in\{0,1,2\}.$

2. **Timestamp transition time**
   For trades occurring at distinct timestamps, record the elapsed time:

   $$\Large \Delta\tau(t_k) = t_k - t_{k-1} \quad \text{where } t_k \neq t_{k-1}$$

3. **Same-timestamp assignment**
   **When multiple trades occur at the same timestamp $t_k$, they all receive the same $\Delta\tau$ value:**

   $$\Large \Delta\tau_{i\to j}(t_k) = \Delta\tau(t_k) = t_k - t_{\text{prev}}$$

   where $t_{\text{prev}}$ is the most recent **distinct** timestamp before $t_k$.

4. **Building the series**
   Collect these durations in chronological order:

$$\Large \{\Delta\tau_{i\to j}(t)\} = \{\Delta\tau_{i\to j}(t_{k_1}),\,\Delta\tau_{i\to j}(t_{k_2}),\,\dots\},$$

   where $k_1, k_2, \dots$ index just the steps with an $i\to j$ transition, and **multiple transitions at the same timestamp get identical $\Delta\tau$ values**.

**Interpretation:**
* $\Delta\tau_{i\to j}(t)$ is "the time since the last distinct timestamp when regime $i$ transitioned to $j$"
* Multiple trades at timestamp $t$ share the same temporal duration
* $\{\Delta\tau_{i\to j}(t)\}$ contains repeated values when transitions cluster at identical timestamps

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
 corr_time_window = 60  # e.g. 1 minute (standard window)
 
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

## **7. Correlation Computation and Interpretation**

### **7.1 Definition of Correlation in SKA Context**

In the SKA framework, we compute **Pearson correlation** between the duration series of paired regime transitions. Specifically, we measure how the **timing patterns** of forward and reverse transitions relate to each other.

**Mathematical Definition:**

For regime pair $(i \leftrightarrow j)$, the correlation coefficient is:

$$ \LARGE \rho_{i\!\leftrightarrow\!j} = \frac{\text{Cov}(\{\Delta\tau_{i\to j}\}, \{\Delta\tau_{j\to i}\})}{\sigma_{\Delta\tau_{i\to j}} \cdot \sigma_{\Delta\tau_{j\to i}}}$$

**Computational Formula (equivalent):**

$$ \LARGE\rho_{i\!\leftrightarrow\!j} = \frac{\sum_{k}(\Delta\tau_{i\to j,k} - \overline{\Delta\tau_{i\to j}})(\Delta\tau_{j\to i,k} - \overline{\Delta\tau_{j\to i}})}{\sqrt{\sum_{k}(\Delta\tau_{i\to j,k} - \overline{\Delta\tau_{i\to j}})^2 \sum_{k}(\Delta\tau_{j\to i,k} - \overline{\Delta\tau_{j\to i}})^2}}$$

where:
- $\{\Delta\tau_{i\to j}\}$ = series of durations for $i \to j$ transitions
- $\{\Delta\tau_{j\to i}\}$ = series of durations for $j \to i$ transitions  
- $\overline{\Delta\tau_{i\to j}}$ = mean duration for $i \to j$ transitions
- $\overline{\Delta\tau_{j\to i}}$ = mean duration for $j \to i$ transitions

**Note:** Both formulas are mathematically identical. The first expresses correlation in terms of covariance and standard deviation, while the second shows the explicit computational steps.

### **7.2 Practical Correlation Computation**

**Step 1: Collect Transition Duration Series**
```python
# Within time window T_corr
delta_tau_i_to_j = []  # e.g., neutral→bull durations
delta_tau_j_to_i = []  # e.g., bull→neutral durations

for transition in window:
    if transition.type == "i→j":
        delta_tau_i_to_j.append(transition.duration)
    elif transition.type == "j→i":
        delta_tau_j_to_i.append(transition.duration)
```

**Step 2: Align Series for Correlation**
The series must be **time-aligned** for meaningful correlation:

```python
# Method 1: Sequential pairing
pairs = []
for idx in range(min(len(delta_tau_i_to_j), len(delta_tau_j_to_i))):
    pairs.append((delta_tau_i_to_j[idx], delta_tau_j_to_i[idx]))

# Method 2: Time-window aggregation  
# Aggregate durations within small time buckets (e.g., 1-minute intervals)
```

**Step 3: Compute Pearson Correlation**
Apply the computational formula from Section 7.1 using your preferred statistics library.

### **7.3 Correlation Interpretation**

| Correlation Value | Market Interpretation | Trading Implication |
|------------------|----------------------|-------------------|
| **+0.8 to +1.0** | **Strong synchronized cycling** | High-confidence trend continuation |
| **+0.3 to +0.8** | **Moderate cycling pattern** | Trend likely but monitor closely |
| **-0.1 to +0.3** | **Weak/random transitions** | Consolidation phase, low predictability |
| **-0.3 to -0.1** | **Slight anti-correlation** | Potential regime change developing |
| **-1.0 to -0.3** | **Strong anti-correlation** | Trend reversal likely |

### **7.4 What Correlation Measures in SKA**

**Positive Correlation ($\rho > 0$):**
- **Synchronized cycling**: When neutral→bull takes long, bull→neutral also takes long
- **Indicates**: Stable trend with predictable oscillation patterns
- **Market behavior**: Consistent information processing rhythm

**Negative Correlation ($\rho < 0$):**
- **Anti-synchronized pattern**: Short neutral→bull followed by long bull→neutral
- **Indicates**: Unstable regime transitions, potential trend exhaustion
- **Market behavior**: Conflicting information signals

**Zero Correlation ($\rho \approx 0$):**
- **Random transition timing**: No predictable relationship between paired transitions
- **Indicates**: Market consolidation or regime uncertainty
- **Market behavior**: Neutral information processing state

### **7.5 Time-Window Considerations**

**Window Size Selection:**
```python
# High-frequency regime detection (immediate patterns)
T_corr_micro = 30      # 30 seconds - ~107 expected transitions
T_corr_short = 60      # 1 minute - ~213 expected transitions

# Medium-term correlation analysis (balanced sensitivity)
T_corr_medium = 120    # 2 minutes - ~426 expected transitions
T_corr_balanced = 180  # 3 minutes - ~639 expected transitions

# Trend analysis (stable patterns)
T_corr_trend = 300     # 5 minutes - ~1,065 expected transitions
T_corr_stable = 600    # 10 minutes - ~2,130 expected transitions
```

**Dynamic Window Adjustment:**
- **High volatility periods**: Use shorter windows
- **Stable markets**: Use longer windows for noise reduction
- **Trend detection**: Start with short windows, extend as correlation stabilizes

### **7.6 Implementation Notes**

**Minimum Sample Requirements:**
- Require at least **10 transitions** of each type for reliable correlation
- **Warning threshold**: Flag correlations computed with <5 samples
- **Skip calculation**: If insufficient data in time window

**Handling Edge Cases:**
```python
def compute_safe_correlation(series_1, series_2):
    if len(series_1) < 5 or len(series_2) < 5:
        return None  # Insufficient data
    
    if np.std(series_1) == 0 or np.std(series_2) == 0:
        return 0.0  # No variation in one series
    
    return pearson_correlation(series_1, series_2)
```

**Real-Time Updates:**
- **Sliding window**: Remove old transitions as new ones arrive
- **Incremental calculation**: Update correlation efficiently without full recalculation
- **Stability monitoring**: Track correlation change rate for regime shift detection

This correlation framework provides the mathematical foundation for detecting and quantifying the **paired regime cycling patterns** that are central to SKA-based trading strategies.