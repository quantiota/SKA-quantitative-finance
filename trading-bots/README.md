# Trading Bots

This folder contains ready-to-run trading algorithms designed to operate on the provided **sample market data** in `/sample_data`.

- `ska_regime_cycling_strategy.py` – SKA-driven regime-based trading bot.
- `markov_chain_strategy.py` – Classic Markov model trading bot for baseline comparison.
- `backtest_utils.py` – Common utilities for backtesting and performance analysis.

## Usage

1. Use any of the precomputed sample CSV files in `/sample_data`.
   - These files already contain entropy and regime information needed for SKA-based trading.
2. Run any script in this folder.

3. Review the output: trade signals, regime plots, and backtest results.


## API Development

Trading bots that **demonstrate robust, repeatable success** on the provided sample data will be candidates for **API integration**.  
The goal: Deploy the most effective trading strategies as real-time, production-ready APIs for use in live trading or analytics pipelines.

- **Criteria:**  
- Consistent performance across multiple datasets  
- Interpretability of signals (SKA regime logic preferred)  
- Ease of API integration

- **Roadmap:**  
1. Benchmark strategies on `/sample_data`.
2. Document winning algorithms and their logic.
3. Develop an API endpoint to serve signals or strategies live.

*If you have a successful bot, reach out to collaborate on the next-generation SKA trading API!*
