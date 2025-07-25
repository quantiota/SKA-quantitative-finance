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

- [x] `data_loader.py` - Primary CSV processing ✅ COMPLETE
- [x] `regime_classifier.py` - Bull/bear/neutral classification ✅ COMPLETE
- [x] `transition_tracker.py` - Δτ time series tracking ✅ COMPLETE
- [x] `correlation_engine.py` - Pearson correlation computation ✅ COMPLETE
- [x] `entropy_probability.py` - Structural probability calculation ✅ COMPLETE
- [x] `signal_generator.py` - Dual-layer correlation+entropy trading signals ✅ COMPLETE
- [x] `ska_strategy.py` - Complete SKA trading strategy ✅ COMPLETE
- [ ] `backtest_engine.py` - Historical simulation framework (integrated into ska_strategy.py)

## Usage

### **Quick Start - Run Complete SKA Trading Strategy**

**1. Execute the complete information-theoretic trading system:**

```bash
cd /home/quant/ska-quantitative-finance/trading_bots/trading_algorithm
python ska_strategy.py
```

This will run the full SKA pipeline and generate comprehensive trading results including dual-layer signal analysis, entropy-based probabilities, and complete performance metrics.

### **Advanced Usage - Individual Components**

**1. Data Analysis and Validation:**
```python
from data_loader import DataLoader

# Load and validate trading data
loader = DataLoader('questdb-query-1751544843847.csv')
data = loader.load_and_validate()
print(f"Loaded {len(data)} trades")
```

**2. SKA Regime Classification:**
```python
from regime_classifier import RegimeClassifier

classifier = RegimeClassifier()
classified_data = classifier.classify_regimes(data)
print(classifier.get_regime_summary())
```

**3. Information-Theoretic Correlation Analysis:**
```python
from correlation_engine import CorrelationEngine
from transition_tracker import TransitionTracker

# Track regime transitions
tracker = TransitionTracker()
transition_data = tracker.track_transitions(classified_data)

# Compute time-windowed correlations
correlator = CorrelationEngine(correlation_time_window=180.0)
correlator.initialize_with_transitions(transition_data, classified_data)
correlation_df = correlator.compute_correlations_over_time()
print(correlator.get_correlation_summary())
```

**4. Entropy-Based Structural Probabilities:**
```python
from entropy_probability import EntropyProbabilityEngine

entropy_engine = EntropyProbabilityEngine()
enhanced_data = entropy_engine.compute_structural_probabilities(classified_data)
print(entropy_engine.get_entropy_summary())
```

**5. Dual-Layer Signal Generation:**
```python
from signal_generator import SKASignalGenerator

signal_gen = SKASignalGenerator(correlation_weight=0.6, entropy_weight=0.4)
signal_gen.initialize(correlation_df, enhanced_data)
signals_df = signal_gen.generate_signals()
print(signal_gen.get_signal_summary())
```

**6. Complete Strategy Backtesting:**
```python
from ska_strategy import SKAStrategy

# Initialize strategy with custom parameters
strategy = SKAStrategy(
    correlation_time_window=180.0,    # 3-minute correlation windows
    correlation_weight=0.6,           # 60% correlation, 40% entropy
    entropy_weight=0.4,
    min_confidence=0.5,               # 50% minimum confidence threshold
    stop_loss_pct=0.01,              # 1% stop loss
    take_profit_pct=0.02             # 2% take profit
)

# Run complete analysis and backtest
strategy.initialize_with_data('questdb-query-1751544843847.csv')
performance = strategy.backtest()
print(strategy.get_performance_summary())
```

### **Parameter Customization Examples**

**Aggressive High-Frequency Setup:**
```python
strategy = SKAStrategy(
    correlation_time_window=60.0,     # 1-minute windows
    min_confidence=0.3,               # Lower confidence threshold
    correlation_weight=0.7,           # Focus on timing patterns
    entropy_weight=0.3
)
```

**Conservative Physics-Based Setup:**
```python
strategy = SKAStrategy(
    correlation_time_window=300.0,    # 5-minute windows
    min_confidence=0.7,               # High confidence only
    correlation_weight=0.3,           # Focus on entropy physics
    entropy_weight=0.7
)
```

---

*Implementation follows the mathematical framework defined in `/mathematical_model/README.md`*