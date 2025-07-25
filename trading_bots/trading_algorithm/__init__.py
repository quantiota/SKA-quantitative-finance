"""
SKA Trading Algorithm - Core Implementation Modules

This package contains the complete implementation of the SKA (Structured Knowledge Accumulation)
trading algorithm based on entropy-driven regime detection and correlation-based signal generation.

Core Modules:
- data_loader: Primary CSV processing and validation
- regime_classifier: Bull/bear/neutral regime classification from price returns
- transition_tracker: Δτ time series tracking for regime transitions
- correlation_engine: Pearson correlation computation for trend pairs
- entropy_probability: Structural probability calculation using entropy dynamics
- signal_generator: Correlation-based trading signal generation
- ska_strategy: Complete SKA trading strategy implementation
- backtest_engine: Historical simulation and performance validation framework

Mathematical Foundation:
Based on the SKA Quantitative Finance framework with entropy-based probability:
P_t = exp(-|ΔH/H_t|) and correlation analysis of paired regime transitions.

Usage:
    from trading_algorithm import SKAStrategy, BacktestEngine
    
    strategy = SKAStrategy()
    engine = BacktestEngine(strategy)
    results = engine.run_backtest('data.csv')
"""

__version__ = "1.0.0"
__author__ = "SKA Quantitative Finance Team"

# Core module imports will be added as modules are implemented
__all__ = [
    'data_loader',
    'regime_classifier', 
    'transition_tracker',
    'correlation_engine',
    'entropy_probability',
    'signal_generator',
    'ska_strategy',
    'backtest_engine'
]