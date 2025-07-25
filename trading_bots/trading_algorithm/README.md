# SKA Trading Algorithm - Core Implementation

This directory contains the core modules for the SKA (Structured Knowledge Accumulation) trading algorithm implementation.

## Module Structure

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

## Implementation Dependencies

```
data_loader.py
    ↓
regime_classifier.py
    ↓
transition_tracker.py
    ↓
correlation_engine.py ←── entropy_probability.py
    ↓
signal_generator.py
    ↓
ska_strategy.py
    ↓
backtest_engine.py
```

## Key Mathematical Components

### Correlation Computation
- **ρ_up**: Correlation between neutral↔bull transition durations
- **ρ_down**: Correlation between neutral↔bear transition durations
- **Time Windows**: Configurable sliding windows (30s, 180s, 600s)

### Entropy-Based Probabilities
- **Structural**: `P_t = exp(-|ΔH/H_t|)`
- **Surface**: `P_{i→j} = ⟨i|Δτ|j⟩/T`
- **Quantum States**: `Ψᵢ = Aᵢ * exp(iHᵢ)`

### Signal Generation Thresholds
- **+0.8 to +1.0**: Strong synchronized cycling → High-confidence continuation
- **+0.3 to +0.8**: Moderate cycling → Trend likely, monitor closely
- **-0.1 to +0.3**: Weak/random → Consolidation phase, low predictability
- **-1.0 to -0.3**: Strong anti-correlation → Trend reversal likely

## Development Status

- [ ] `data_loader.py` - Primary CSV processing
- [ ] `regime_classifier.py` - Bull/bear/neutral classification  
- [ ] `transition_tracker.py` - Δτ time series tracking
- [ ] `correlation_engine.py` - Pearson correlation computation
- [ ] `entropy_probability.py` - Structural probability calculation
- [ ] `signal_generator.py` - Correlation-based trading signals
- [ ] `ska_strategy.py` - Complete SKA trading strategy
- [ ] `backtest_engine.py` - Historical simulation framework

## Usage Example

```python
from trading_algorithm import (
    DataLoader, RegimeClassifier, TransitionTracker,
    CorrelationEngine, SignalGenerator, SKAStrategy, BacktestEngine
)

# Load and process data
loader = DataLoader('primary_data.csv')
data = loader.load_and_validate()

# Initialize strategy components
classifier = RegimeClassifier()
tracker = TransitionTracker()
correlator = CorrelationEngine(time_window=180)
signals = SignalGenerator(correlator)

# Create and run strategy
strategy = SKAStrategy(classifier, tracker, correlator, signals)
engine = BacktestEngine(strategy)
results = engine.run_backtest(data)
```

---

*Implementation follows the mathematical framework defined in `/mathematical_model/README.md`*