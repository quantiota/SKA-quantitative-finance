# Sample Market Data

This folder contains primary CSV files with raw trade data used for SKA-quantitative-finance research and analysis.

## Contents

- High-frequency trade/tick data samples (e.g., `XRPUSDT_2024-07-06.csv`)
- **Each CSV file includes the following columns:**
  - `trade_id` – unique identifier for each trade
  - `timestamp` – ISO8601 datetime of the trade
  - `price` – executed trade price (float)
  - `volume` – traded amount (float)
  - `entropy` – computed entropy value for the trade (float)
  - `side` – trade direction (`buy` or `sell`)
  - `buyer_id` – (if available) ID of buyer participant
  - `seller_id` – (if available) ID of seller participant

  *(Some columns may be absent or additional columns may appear, depending on the dataset.)*

## Purpose

- Enable other researchers and quants to **replicate key figures** (entropy trajectories, regime transitions, probability bands) published in this repository
- Allow independent analysis and benchmarking using the same datasets
- Facilitate open scientific discussion and peer review

## Usage

- You are free to use these sample data files for academic research, non-commercial analysis, or backtesting.
- Please cite the main repository if publishing results or derivative work.

## Note

- **The real-time SKA learning code is not included in this folder.**
- Only the non real-time (batch) code is available in the [arXiv repository](https://github.com/quantiota/Arxiv).
- **Coming soon:** Real-time SKA learning will be available via an API, allowing researchers and quants to test SKA analytics on their own data streams.

---
