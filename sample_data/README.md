# Sample Market Data

This folder contains primary CSV files with raw trade data used for SKA-quantitative-finance research and analysis.

## Contents

- High-frequency trade/tick data samples (e.g., `questdb-query-1752003744108`)
- **Each CSV file includes the following columns:**
  - `trade_id` – unique identifier for each trade
  - `timestamp` – ISO8601 datetime of the trade
  - `price` – executed trade price (float)
  - `quantity` – traded amount (float)
  - `entropy` – computed entropy value for the trade (float)
  - `buyer_order_id` – (if available) ID of buyer participant
  - `seller_order_id` – (if available) ID of seller participant

  *(Some columns may be absent or additional columns may appear, depending on the dataset.)*

## Purpose

- Enable other researchers and quants to **replicate key figures** (entropy trajectories, regime transitions, probability bands) published in this repository
- Allow independent analysis and benchmarking using the same datasets
- With these sample data files, quantitative researchers can test and develop their own entropy-based BUY/SELL strategies, regime detection methods, or risk models—using the same high-frequency trade data and entropy measures analyzed in this project.
- Facilitate open scientific discussion and peer review


> Unlike classical models, SKA reveals hidden regime cycles directly from entropy dynamics, not from price trends.



## Example Insight: Regime Cycling in Trends

The following figure, generated with SKA entropy learning from the file `questdb-query-1752003744108.csv`, illustrates a key market microstructure insight:

Uptrends are not composed of persistent `bull→bull` transitions. Instead, they are driven by rapid alternation between `neutral→bull` and `bull→neutral` transitions.

This reveals that market trends are fundamentally constructed from `paired` regime transitions—specifically, the rapid cycling between `neutral→bull` and `bull→neutral` (for uptrends), or `neutral→bear` and `bear→neutral` (for downtrends). These paired transitions, rather than persistence in a single state, are the true information-processing units of market trend.

> **Breakthrough:**
> The regime cycling structure shown below is extracted purely from entropy learning—not from price or state aggregation.
> The alignment between these entropy-driven regime transitions and actual price movements offers *concrete visual proof* that markets operate under hidden informational laws, not just random walks.
> This provides quants and researchers with a glimpse into the next generation of market analysis.

![SKA Transition Probability Figure](probability_with_price.png)

**Interpretation:**

* During price uptrends (see bottom plot), the transitions with highest frequency are **neutral→bull** and **bull→neutral**.
* **Persistent “bull→bull” transitions are rare!** (As confirmed in the legend.)
* SKA’s entropy-driven, trade-by-trade visualization makes this market regime cycling explicit—revealing a universal information-processing law of market behavior.
* The same cycling phenomenon is observed during downtrends, where rapid alternation between **neutral→bear** and **bear→neutral** transitions dominates the price movement.

> **Empirical foundation:**
> This result powerfully reinforces the SKA entropy definition.
> The learned regime transitions—derived purely from entropy dynamics—align so closely with price evolution that SKA’s information-theoretic approach clearly captures the *true underlying structure* of market behavior.
> In other words, SKA’s entropy is not just a mathematical construct—it describes real, observable laws that govern how markets process information and evolve.


## Background: Markov Chain and Regime-Switching Models

The standard approach in quantitative finance for modeling trends and market regimes is the **Markov chain/regime-switching model**. The canonical reference is:

> Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press.

These models define an uptrend as a persistent run of `bull` states, detected by high self-transition probabilities (`P(bull→bull)`). Most classical quant analysis, including academic and industry research, follows this paradigm.

However, it is important to note that classical Markov chain and regime-switching models operate at a **coarser temporal resolution**—they characterize regime persistence and transitions over aggregated sequences or time windows.

**In contrast, SKA works at a much finer, event-by-event resolution:**  
By tracking entropy and regime transitions at the level of each trade (or event), SKA reveals market microstructure dynamics and `paired` regime cycling that remain invisible to traditional Markov/state-count approaches.




## Usage

- You are free to use these sample data files for academic research, non-commercial analysis, or backtesting.
- Please cite the main repository if publishing results or derivative work.

## Note

- **The real-time SKA learning code is not included in this folder.**
- Only the non real-time (batch) code is available in the [arXiv repository](https://github.com/quantiota/Arxiv).
- **Coming soon:** Real-time SKA learning will be available via an API, allowing researchers and quants to test SKA analytics on their own data streams



