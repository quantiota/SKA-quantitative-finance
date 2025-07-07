
# SKA Quantitative Finance

## Real-Time Quantitative Research with Structured Knowledge Accumulation

**SKA Quantitative Finance** is a real-time quantitative research project for high-frequency financial data, powered by the **Structured Knowledge Accumulation (SKA)** framework ([arXiv:2503.13942](https://arxiv.org/abs/2503.13942), [arXiv:2504.03214](https://arxiv.org/abs/2504.03214)).
SKA provides a fundamentally new approach to learning and market analysis based on **entropy-driven, forward-only, biologically plausible algorithms**. Instead of relying on classical backpropagation, SKA uses a **layer-wise, entropy-minimizing process** to structure knowledge and decision-making over time, enabling deep insight into both neural networks and real-time market microstructure.

---

## Project Description

SKA Quantitaive Finance processes tick-level trade data (e.g., from cryptocurrency markets) to compute **entropy trajectories** and classify each trade into one of three market regimes: **bull**, **bear**, or **neutral**.
By continuously measuring entropy (market disorder) and aligning it with observed decision probabilities, SKA Quantitative Finance offers a dynamic and granular view of how market structure evolves at the smallest time scales.

* **Structured Knowledge Accumulation (SKA)**: A neural and market learning paradigm where entropy is minimized locally, layer-by-layer, without backpropagation or global optimization.
* **Entropy Trajectories**: Quantitative time series capturing the evolution of uncertainty (entropy) at every trade event.
* **Regime Segmentation**: Each trade is classified as bull (price up), bear (price down), or neutral (no change), based on real-time returns and entropy flow.

For full theoretical details, see the [first SKA paper](https://arxiv.org/abs/2503.13942) and [second SKA paper](https://arxiv.org/abs/2504.03214).

---

## Key Features

* **Real-Time Data Ingestion & Preprocessing**: Processes high-frequency trade data, normalizes time sequences, and calculates per-trade entropy and regime.
* **Entropy Computation and Regime Classification**: Implements SKA's entropy and structured knowledge equations for both neural and market data.
* **Export of Normalized Trajectories**: Outputs separate CSV files for each regime (bull, bear, neutral) containing trade sequence, price, and entropy.
* **Entropy Transition Probability Modeling**: Quantifies regime change likelihoods using entropy-based transition probabilities.

---
## SKA vs. Markov Regime Models

For a detailed exploration of how SKA’s entropy-based transition probabilities reveal the hidden information-theoretic basis for constant regime switching—mirroring and explaining the empirical success of Markov chain models in finance—see the [SKA vs Markov](https://github.com/quantiota/SKA-quantitative-finance/blob/main/ska_vs_markov.md) document included in this repository.

---

## Installation

**Requirements:**

* Python 3.x
* `pandas`, `numpy`, `matplotlib`

Install dependencies via pip:

```bash
pip install pandas numpy matplotlib
```



---
## Usage

**1. Data Preparation:**
Provide a high-frequency tick/trade CSV file (see script comments for expected columns: `trade_id`, `timestamp`, `price`, `entropy`, etc.).

**2. Entropy and Regime Trajectories:**
Run:

```bash
python export.py
```

This will:

* Load and preprocess data
* Compute entropy, price return, and regime
* Output:
  * `transition_bull_bull.csv`, ..., `transition_neutral_neutral.csv` (all 9 transition CSVs)
* Print summary stats and generate entropy plots

**3. Transition Probability and Price Visualization:**
Run:

```bash
python probability_with_price.py
```

This will:

* Combine the 9 transition CSVs
* Compute regime transitions and entropy-based transition probabilities for every trade
* Produce: `probability_with_price.png` and `.svg`
  (Scatter plot of transition probabilities, color/shape-coded by transition, overlaid with price)

---

## Outputs

* `transition_bull_bull.csv`, ..., `transition_neutral_neutral.csv`
* `probability_with_price.png`, `probability_with_price.svg`



---

## Example Insights

* **Neutral regime dominance:** Most trades are neutral (often 75–80%), reflecting market consolidation phases.
* **Universal transition probabilities:** Certain transitions (like neutral→neutral) occur with universal, information-theoretically constant probabilities—uncovered only by SKA.
* **Entropy transitions highlight volatility clusters:** Sudden shifts in regime and drops in transition probability reveal where price and uncertainty spike together.
* **Direct LaTeX integration:** Reproducible, publication-quality visuals for scientific communication.

---

## Citing SKA

If you use SKA Quantitative Finance, please cite:

* Bouarfa Mahi.
  **Structured Knowledge Accumulation: An Autonomous Framework for Layer-Wise Entropy Reduction in Neural Learning**
  [arXiv:2503.13942](https://arxiv.org/abs/2503.13942)
* Bouarfa Mahi.
  **Structured Knowledge Accumulation: The Principle of Entropic Least Action in Forward-Only Neural Learning**
  [arXiv:2504.03214](https://arxiv.org/abs/2504.03214)

```
@article{mahi2025ska1,
  title={Structured Knowledge Accumulation: An Autonomous Framework for Layer-Wise Entropy Reduction in Neural Learning},
  author={Mahi, Bouarfa},
  journal={arXiv preprint arXiv:2503.13942},
  year={2025}
}
@article{mahi2025ska2,
  title={Structured Knowledge Accumulation: The Principle of Entropic Least Action in Forward-Only Neural Learning},
  author={Mahi, Bouarfa},
  journal={arXiv preprint arXiv:2504.03214},
  year={2025}
}
```

---

## License and Contribution

Open for research and academic use (MIT License).
Contributions are welcome! Please open an issue or pull request to suggest enhancements or extensions.

---

*SKA Quantitative Finance bridges the theory of Structured Knowledge Accumulation (SKA) with real-time quantitative market research — enabling new scientific discovery at the intersection of AI, entropy, and financial dynamics.*