"""
SKA Trading Algorithm - Complete SKA Strategy Implementation

Complete SKA regime cycling strategy integrating all components: regime classification,
transition tracking, correlation analysis, entropy probabilities, and signal generation.

Author: SKA Quantitative Finance Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Import SKA components
from data_loader import DataLoader
from regime_classifier import RegimeClassifier
from transition_tracker import TransitionTracker
from correlation_engine import CorrelationEngine
from entropy_probability import EntropyProbabilityEngine
from signal_generator import SKASignalGenerator, SignalDirection, SignalStrength

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position representation."""
    direction: str  # 'long', 'short', 'neutral'
    size: float
    entry_price: float
    entry_time: datetime
    entry_signal: str
    entry_confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_pnl: float = 0.0


@dataclass
class Trade:
    """Completed trade record."""
    direction: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    entry_signal: str
    exit_signal: str
    pnl: float
    pnl_pct: float
    duration_seconds: float
    entry_confidence: float
    exit_confidence: float


class SKAStrategy:
    """
    Complete SKA trading strategy implementation.
    
    Revolutionary information-theoretic trading system combining:
    1. Regime classification (information states)
    2. Transition tracking (Δτ time series)
    3. Correlation analysis (timing patterns) 
    4. Entropy probabilities (structural laws)
    5. Dual-layer signal generation
    
    First trading system based on market information physics.
    """
    
    def __init__(self,
                 correlation_time_window: float = 180.0,
                 correlation_weight: float = 0.6,
                 entropy_weight: float = 0.4,
                 signal_threshold: float = 0.3,
                 min_confidence: float = 0.5,
                 position_size: float = 1.0,
                 stop_loss_pct: float = 0.01,
                 take_profit_pct: float = 0.02):
        """
        Initialize SKA strategy.
        
        Args:
            correlation_time_window (float): Time window for correlation analysis (seconds)
            correlation_weight (float): Weight for correlation signals (0-1)
            entropy_weight (float): Weight for entropy signals (0-1)
            signal_threshold (float): Minimum signal threshold for trading
            min_confidence (float): Minimum confidence for position entry
            position_size (float): Position size (normalized)
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
        """
        # Strategy parameters
        self.correlation_time_window = correlation_time_window
        self.correlation_weight = correlation_weight
        self.entropy_weight = entropy_weight
        self.signal_threshold = signal_threshold
        self.min_confidence = min_confidence
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # SKA components
        self.data_loader = DataLoader("")
        self.regime_classifier = RegimeClassifier()
        self.transition_tracker = TransitionTracker()
        self.correlation_engine = CorrelationEngine(correlation_time_window)
        self.entropy_engine = EntropyProbabilityEngine()
        self.signal_generator = SKASignalGenerator(correlation_weight, entropy_weight, signal_threshold)
        
        # Strategy state
        self.current_position = None
        self.trades = []
        self.signals_history = []
        self.performance_metrics = {}
        
        # Data storage
        self.raw_data = None
        self.classified_data = None
        self.transition_data = None
        self.correlation_data = None
        self.entropy_enhanced_data = None
        self.signals_data = None
        
    def initialize_with_data(self, data_path: str) -> None:
        """
        Initialize strategy with trading data.
        
        Args:
            data_path (str): Path to primary CSV data file
        """
        logger.info(f"Initializing SKA strategy with data: {data_path}")
        
        # Load and validate data
        self.data_loader = DataLoader(data_path)
        self.raw_data = self.data_loader.load_and_validate()
        
        # Process through complete SKA pipeline
        self._run_ska_pipeline()
        
        logger.info("SKA strategy initialization completed")
    
    def _run_ska_pipeline(self) -> None:
        """Run complete SKA analysis pipeline."""
        logger.info("Running complete SKA analysis pipeline")
        
        # 1. Regime classification
        self.classified_data = self.regime_classifier.classify_regimes(self.raw_data)
        
        # 2. Transition tracking
        self.transition_data = self.transition_tracker.track_transitions(self.classified_data)
        
        # 3. Correlation analysis
        self.correlation_engine.initialize_with_transitions(self.transition_data, self.classified_data)
        self.correlation_data = self.correlation_engine.compute_correlations_over_time()
        
        # 4. Entropy probability analysis
        self.entropy_enhanced_data = self.entropy_engine.compute_structural_probabilities(self.classified_data)
        
        # 5. Signal generation
        self.signal_generator.initialize(self.correlation_data, self.entropy_enhanced_data)
        self.signals_data = self.signal_generator.generate_signals()
        
        logger.info("SKA pipeline execution completed")
    
    def backtest(self) -> Dict:
        """
        Run complete backtest simulation.
        
        Returns:
            Dict: Comprehensive backtest results
        """
        logger.info("Starting SKA strategy backtest")
        
        if self.signals_data is None:
            raise ValueError("Must run SKA pipeline first")
        
        # Reset state
        self.current_position = None
        self.trades = []
        self.signals_history = []
        
        # Simulate trading on signals
        for _, signal in self.signals_data.iterrows():
            self._process_signal(signal)
        
        # Close any remaining position
        if self.current_position is not None:
            self._close_position(self.signals_data.iloc[-1], "end_of_data")
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        logger.info(f"Backtest completed: {len(self.trades)} trades executed")
        return self.performance_metrics
    
    def _process_signal(self, signal: pd.Series) -> None:
        """
        Process individual trading signal.
        
        Args:
            signal (pd.Series): Signal data for specific timestamp
        """
        timestamp = signal['timestamp']
        signal_direction = signal['final_signal']
        signal_strength = signal['final_strength']
        confidence = signal['final_confidence']
        
        # Store signal in history
        self.signals_history.append({
            'timestamp': timestamp,
            'signal': signal_direction,
            'strength': signal_strength,
            'confidence': confidence,
            'action_taken': 'none'
        })
        
        # Get current price (use entropy data for price context)
        current_price = self._get_price_at_timestamp(timestamp)
        if current_price is None:
            return
        
        # Update current position P&L
        if self.current_position is not None:
            self._update_position_pnl(current_price)
        
        # Check for position exits first
        if self.current_position is not None:
            exit_action = self._check_exit_conditions(signal, current_price)
            if exit_action:
                self._close_position(signal, exit_action)
                self.signals_history[-1]['action_taken'] = f'exit_{exit_action}'
                return
        
        # Check for new position entries
        if self.current_position is None and confidence >= self.min_confidence:
            entry_action = self._check_entry_conditions(signal)
            if entry_action:
                self._open_position(signal, entry_action, current_price)
                self.signals_history[-1]['action_taken'] = f'entry_{entry_action}'
    
    def _get_price_at_timestamp(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Get price at specific timestamp from entropy enhanced data."""
        if self.entropy_enhanced_data is None:
            return None
        
        # Ensure timezone consistency
        target_time = pd.Timestamp(timestamp)
        entropy_timestamps = pd.to_datetime(self.entropy_enhanced_data['timestamp'])
        
        # Remove timezone info for comparison
        if target_time.tz is not None:
            target_time = target_time.tz_localize(None)
        if entropy_timestamps.dt.tz is not None:
            entropy_timestamps = entropy_timestamps.dt.tz_localize(None)
        
        # Find closest timestamp in entropy data
        time_diffs = abs(entropy_timestamps - target_time)
        closest_idx = time_diffs.idxmin()
        
        # Return price if within reasonable tolerance (30 seconds)
        if time_diffs.iloc[closest_idx] <= pd.Timedelta(seconds=30):
            return self.entropy_enhanced_data.loc[closest_idx, 'price']
        
        return None
    
    def _update_position_pnl(self, current_price: float) -> None:
        """Update current position P&L."""
        if self.current_position is None:
            return
        
        if self.current_position.direction == 'long':
            self.current_position.current_pnl = (current_price - self.current_position.entry_price) * self.current_position.size
        elif self.current_position.direction == 'short':
            self.current_position.current_pnl = (self.current_position.entry_price - current_price) * self.current_position.size
    
    def _check_exit_conditions(self, signal: pd.Series, current_price: float) -> Optional[str]:
        """
        Check if current position should be exited.
        
        Returns:
            Optional[str]: Exit reason or None
        """
        if self.current_position is None:
            return None
        
        position = self.current_position
        signal_direction = signal['final_signal']
        confidence = signal['final_confidence']
        
        # Stop loss check
        if position.stop_loss is not None:
            if ((position.direction == 'long' and current_price <= position.stop_loss) or
                (position.direction == 'short' and current_price >= position.stop_loss)):
                return 'stop_loss'
        
        # Take profit check
        if position.take_profit is not None:
            if ((position.direction == 'long' and current_price >= position.take_profit) or
                (position.direction == 'short' and current_price <= position.take_profit)):
                return 'take_profit'
        
        # Signal-based exit (reversal or strong opposing signal)
        if confidence >= self.min_confidence:
            if ((position.direction == 'long' and signal_direction in ['strong_bear', 'moderate_bear', 'reversal_imminent']) or
                (position.direction == 'short' and signal_direction in ['strong_bull', 'moderate_bull', 'reversal_imminent'])):
                return 'signal_reversal'
        
        # Consolidation exit (low confidence in current direction)
        if signal_direction == 'consolidation' and confidence > 0.7:
            return 'consolidation'
        
        return None
    
    def _check_entry_conditions(self, signal: pd.Series) -> Optional[str]:
        """
        Check if new position should be opened.
        
        Returns:
            Optional[str]: Position direction or None
        """
        signal_direction = signal['final_signal']
        signal_strength = signal['final_strength']
        confidence = signal['final_confidence']
        
        # Require minimum signal strength for entry
        if signal_strength in ['very_weak', 'weak']:
            return None
        
        # Bull signals
        if signal_direction in ['strong_bull', 'moderate_bull'] and confidence >= self.min_confidence:
            return 'long'
        
        # Bear signals  
        if signal_direction in ['strong_bear', 'moderate_bear'] and confidence >= self.min_confidence:
            return 'short'
        
        # Weak directional signals with high confidence
        if signal_direction == 'weak_bull' and confidence >= 0.8:
            return 'long'
        if signal_direction == 'weak_bear' and confidence >= 0.8:
            return 'short'
        
        return None
    
    def _open_position(self, signal: pd.Series, direction: str, price: float) -> None:
        """Open new trading position."""
        # Calculate stop loss and take profit
        if direction == 'long':
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
        else:  # short
            stop_loss = price * (1 + self.stop_loss_pct)
            take_profit = price * (1 - self.take_profit_pct)
        
        self.current_position = Position(
            direction=direction,
            size=self.position_size,
            entry_price=price,
            entry_time=signal['timestamp'],
            entry_signal=signal['final_signal'],
            entry_confidence=signal['final_confidence'],
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        logger.debug(f"Opened {direction} position at {price:.4f} (confidence: {signal['final_confidence']:.3f})")
    
    def _close_position(self, signal: pd.Series, exit_reason: str) -> None:
        """Close current trading position."""
        if self.current_position is None:
            return
        
        # Get exit price
        exit_price = self._get_price_at_timestamp(signal['timestamp'])
        if exit_price is None:
            exit_price = self.current_position.entry_price  # Fallback
        
        # Calculate P&L
        if self.current_position.direction == 'long':
            pnl = (exit_price - self.current_position.entry_price) * self.current_position.size
        else:  # short
            pnl = (self.current_position.entry_price - exit_price) * self.current_position.size
        
        pnl_pct = pnl / (self.current_position.entry_price * self.current_position.size) * 100
        duration = (signal['timestamp'] - self.current_position.entry_time).total_seconds()
        
        # Create trade record
        trade = Trade(
            direction=self.current_position.direction,
            size=self.current_position.size,
            entry_price=self.current_position.entry_price,
            exit_price=exit_price,
            entry_time=self.current_position.entry_time,
            exit_time=signal['timestamp'],
            entry_signal=self.current_position.entry_signal,
            exit_signal=exit_reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_seconds=duration,
            entry_confidence=self.current_position.entry_confidence,
            exit_confidence=signal.get('final_confidence', 0.0)
        )
        
        self.trades.append(trade)
        self.current_position = None
        
        logger.debug(f"Closed {trade.direction} position: {trade.pnl:+.4f} ({trade.pnl_pct:+.2f}%) in {trade.duration_seconds:.0f}s")
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate comprehensive performance metrics."""
        # Signal analysis (always available)
        signals_taken = len([s for s in self.signals_history if s['action_taken'] != 'none'])
        total_signals = len(self.signals_history)
        signal_efficiency = signals_taken / total_signals if total_signals > 0 else 0
        
        if len(self.trades) == 0:
            self.performance_metrics = {
                'total_trades': 0,
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'signal_efficiency': signal_efficiency,
                'signals_generated': total_signals,
                'signals_acted_upon': signals_taken,
                'avg_trade_duration_seconds': 0.0,
                'avg_entry_confidence': 0.0,
                'strategy_parameters': {
                    'correlation_window': self.correlation_time_window,
                    'correlation_weight': self.correlation_weight,
                    'entropy_weight': self.entropy_weight,
                    'min_confidence': self.min_confidence,
                    'position_size': self.position_size,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct
                }
            }
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_pnl_pct = sum(trade.pnl_pct for trade in self.trades)
        
        # Win/loss analysis
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Drawdown analysis
        cumulative_pnl = np.cumsum([t.pnl for t in self.trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(self.trades) > 1:
            returns = [t.pnl_pct for t in self.trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Signal analysis
        signals_taken = len([s for s in self.signals_history if s['action_taken'] != 'none'])
        total_signals = len(self.signals_history)
        signal_efficiency = signals_taken / total_signals if total_signals > 0 else 0
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'signal_efficiency': signal_efficiency,
            'signals_generated': total_signals,
            'signals_acted_upon': signals_taken,
            'avg_trade_duration_seconds': np.mean([t.duration_seconds for t in self.trades]),
            'avg_entry_confidence': np.mean([t.entry_confidence for t in self.trades]),
            'strategy_parameters': {
                'correlation_window': self.correlation_time_window,
                'correlation_weight': self.correlation_weight,
                'entropy_weight': self.entropy_weight,
                'min_confidence': self.min_confidence,
                'position_size': self.position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }
        }
    
    def get_performance_summary(self) -> str:
        """Get formatted performance summary."""
        if not self.performance_metrics:
            return "No backtest results available"
        
        metrics = self.performance_metrics
        
        summary = f"""
SKA Strategy Performance Summary
================================
Strategy: Information-Theoretic Dual-Layer Analysis
Correlation Window: {metrics['strategy_parameters']['correlation_window']:.0f}s
Weighting: Correlation {metrics['strategy_parameters']['correlation_weight']:.1f} | Entropy {metrics['strategy_parameters']['entropy_weight']:.1f}

Trading Performance:
  Total Trades: {metrics['total_trades']}
  Total P&L: {metrics['total_pnl']:+.4f}
  Total P&L %: {metrics['total_pnl_pct']:+.2f}%
  Win Rate: {metrics['win_rate']:.1%}
  Average Win: {metrics['avg_win']:+.4f}
  Average Loss: {metrics['avg_loss']:+.4f}
  Profit Factor: {metrics['profit_factor']:.2f}
  Max Drawdown: {metrics['max_drawdown']:.4f}
  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}

Signal Analysis:
  Signals Generated: {metrics['signals_generated']}
  Signals Acted Upon: {metrics['signals_acted_upon']}
  Signal Efficiency: {metrics['signal_efficiency']:.1%}
  Avg Entry Confidence: {metrics['avg_entry_confidence']:.3f}
  Avg Trade Duration: {metrics['avg_trade_duration_seconds']:.0f} seconds

Risk Management:
  Position Size: {metrics['strategy_parameters']['position_size']:.2f}
  Stop Loss: {metrics['strategy_parameters']['stop_loss_pct']:.1%}
  Take Profit: {metrics['strategy_parameters']['take_profit_pct']:.1%}
  Min Confidence: {metrics['strategy_parameters']['min_confidence']:.1%}
        """
        
        return summary.strip()
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get all trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'direction': trade.direction,
                'size': trade.size,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_signal': trade.entry_signal,
                'exit_signal': trade.exit_signal,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'duration_seconds': trade.duration_seconds,
                'entry_confidence': trade.entry_confidence,
                'exit_confidence': trade.exit_confidence
            })
        
        return pd.DataFrame(trades_data)


def run_ska_strategy(data_path: str, **strategy_params) -> Tuple[SKAStrategy, Dict]:
    """
    Convenience function to run complete SKA strategy.
    
    Args:
        data_path (str): Path to trading data
        **strategy_params: Strategy configuration parameters
        
    Returns:
        Tuple[SKAStrategy, Dict]: (strategy_instance, performance_metrics)
    """
    strategy = SKAStrategy(**strategy_params)
    strategy.initialize_with_data(data_path)
    performance = strategy.backtest()
    
    return strategy, performance


if __name__ == "__main__":
    # Test complete SKA strategy
    file_path = "questdb-query-1751544843847.csv"
    
    try:
        # Run SKA strategy with default parameters
        strategy = SKAStrategy(
            correlation_time_window=180.0,
            correlation_weight=0.6,
            entropy_weight=0.4,
            min_confidence=0.5,
            position_size=1.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.02
        )
        
        strategy.initialize_with_data(file_path)
        performance = strategy.backtest()
        
        # Display results
        print(strategy.get_performance_summary())
        
        # Show sample trades
        trades_df = strategy.get_trades_dataframe()
        if len(trades_df) > 0:
            print(f"\nSample Trades:")
            display_cols = ['direction', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'duration_seconds', 'entry_confidence']
            print(trades_df[display_cols].head().to_string(index=False))
        else:
            print(f"\nNo trades executed (signals may not have met confidence threshold)")
        
        print(f"\n🚀 BREAKTHROUGH: First complete information-theoretic trading strategy executed!")
        print(f"Revolutionary dual-layer SKA system operational!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()