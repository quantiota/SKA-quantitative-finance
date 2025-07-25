"""
SKA Trading Algorithm - Core Implementation Modules

This package contains the complete implementation of the SKA (Structured Knowledge Accumulation)
trading algorithm - the world's first information-theoretic trading system based on entropy dynamics
and dual-layer correlation analysis.

Core Modules:
- data_loader: Primary CSV processing and validation
- regime_classifier: Bull/bear/neutral regime classification from price returns
- transition_tracker: Δτ time series tracking for regime transitions
- correlation_engine: Pearson correlation computation for trend pairs
- entropy_probability: Structural probability calculation using entropy dynamics
- signal_generator: Dual-layer correlation and entropy-based trading signals
- ska_strategy: Complete SKA trading strategy with integrated backtesting

Mathematical Foundation:
Revolutionary information-theoretic approach using:
- Structural probabilities: P_t = exp(-|ΔH/H_t|)
- Quantum state formulation: Ψᵢ = Aᵢ * exp(iHᵢ)
- Correlation analysis of paired regime transitions
- Dual-layer signal generation combining timing patterns with entropy physics

Usage:
    from trading_algorithm.ska_strategy import SKAStrategy
    
    strategy = SKAStrategy()
    strategy.initialize_with_data('data.csv')
    performance = strategy.backtest()
    print(strategy.get_performance_summary())
"""

__version__ = "1.0.0"
__author__ = "SKA Quantitative Finance Team"

# Import completed modules
from .data_loader import DataLoader
from .regime_classifier import RegimeClassifier
from .transition_tracker import TransitionTracker
from .correlation_engine import CorrelationEngine
from .entropy_probability import EntropyProbabilityEngine
from .signal_generator import SKASignalGenerator
from .ska_strategy import SKAStrategy

__all__ = [
    'DataLoader',
    'RegimeClassifier', 
    'TransitionTracker',
    'CorrelationEngine',
    'EntropyProbabilityEngine',
    'SKASignalGenerator',
    'SKAStrategy'
]