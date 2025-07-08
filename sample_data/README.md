# Sample Market Data

This folder contains primary CSV files with raw trade data used for SKA-quantitative-finance research and analysis.

## Contents

- High-frequency trade/tick data samples (e.g., `XRPUSDT_2024-07-06.csv`)
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


## Usage

- You are free to use these sample data files for academic research, non-commercial analysis, or backtesting.
- Please cite the main repository if publishing results or derivative work.

## Note

- **The real-time SKA learning code is not included in this folder.**
- Only the non real-time (batch) code is available in the [arXiv repository](https://github.com/quantiota/Arxiv).
- **Coming soon:** Real-time SKA learning will be available via an API, allowing researchers and quants to test SKA analytics on their own data streams.

---

> **For Industry Partners and Quantitative Finance Teams:**  
> The SKA-quantitative-finance project introduces a novel, information-theoretic framework for real-time regime detection and market analysis, grounded in entropy minimization and biologically plausible learning. Unlike traditional models, SKA uncovers hidden structure and regime dynamics directly from high-frequency data—enabling new insights for trading, risk management, and alpha generation. We welcome inquiries from financial institutions, quant teams, and technology partners interested in exploring or validating SKA’s edge in practical market environments.
