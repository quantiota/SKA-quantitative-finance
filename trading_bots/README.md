# Trading Bots

This folder contains ready-to-run trading algorithms designed to operate on the provided **sample market data** in `/sample_data`.


### Core Processing Modules
- **`data_loader.py`** - Primary CSV data processing and validation
- **`regime_classifier.py`** - Bull/bear/neutral regime classification from price returns
- **`transition_tracker.py`** - Δτ time series tracking for all regime transitions

### Mathematical Framework
- **`correlation_engine.py`** - Pearson correlation computation for trend pairs
- **`entropy_probability.py`** - Structural probability calculation using entropy dynamics

### Trading Strategy
- **`signal_generator.py`** - Correlation-based trading signal generation
- **`ska_strategy.py`** - Complete SKA trading strategy implementation
- **`backtest_engine.py`** - Historical simulation and performance validation


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
