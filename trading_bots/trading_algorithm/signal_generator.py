"""
SKA Trading Algorithm - Signal Generator Module

Dual-layer correlation and entropy-based trading signal generation for the SKA quantitative finance framework.
Combines surface correlation patterns with structural entropy probabilities to create revolutionary information-theoretic trading signals.

Author: SKA Quantitative Finance Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength enumeration."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate" 
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalDirection(Enum):
    """Signal direction enumeration."""
    STRONG_BULL = "strong_bull"
    MODERATE_BULL = "moderate_bull"
    WEAK_BULL = "weak_bull"
    NEUTRAL = "neutral"
    CONSOLIDATION = "consolidation"
    WEAK_BEAR = "weak_bear"
    MODERATE_BEAR = "moderate_bear"
    STRONG_BEAR = "strong_bear"
    REVERSAL_IMMINENT = "reversal_imminent"


class SKASignalGenerator:
    """
    SKA signal generator combining correlation and entropy analysis.
    
    Revolutionary dual-layer approach:
    1. Surface Layer: Correlation patterns (ρ_up, ρ_down) reveal timing relationships
    2. Structural Layer: Entropy probabilities (P_t) reveal fundamental transition physics
    
    Combined Analysis: First information-theoretic trading signals in history
    """
    
    def __init__(self, 
                 correlation_weight: float = 0.6,
                 entropy_weight: float = 0.4,
                 signal_threshold: float = 0.3):
        """
        Initialize SKA signal generator.
        
        Args:
            correlation_weight (float): Weight for correlation signals (0-1)
            entropy_weight (float): Weight for entropy signals (0-1)  
            signal_threshold (float): Minimum threshold for signal generation
        """
        if abs(correlation_weight + entropy_weight - 1.0) > 1e-6:
            raise ValueError("Correlation and entropy weights must sum to 1.0")
            
        self.correlation_weight = correlation_weight
        self.entropy_weight = entropy_weight
        self.signal_threshold = signal_threshold
        
        # Data storage
        self.correlation_data = None
        self.entropy_data = None
        self.signals = []
        self.current_signal = None
        
        # Signal statistics
        self.signal_statistics = {}
        
    def initialize(self, correlation_df: pd.DataFrame, entropy_enhanced_data: pd.DataFrame) -> None:
        """
        Initialize generator with correlation and entropy data.
        
        Args:
            correlation_df (pd.DataFrame): Time series correlation analysis
            entropy_enhanced_data (pd.DataFrame): Data with structural probabilities
        """
        self.correlation_data = correlation_df.copy()
        self.entropy_data = entropy_enhanced_data.copy()
        
        logger.info(f"Signal generator initialized with {len(correlation_df)} correlation periods and {len(entropy_enhanced_data)} entropy calculations")
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate complete signal time series combining both analysis layers.
        
        Returns:
            pd.DataFrame: Complete trading signals with timestamps
        """
        if self.correlation_data is None or self.entropy_data is None:
            raise ValueError("Must initialize with correlation and entropy data first")
        
        logger.info("Generating dual-layer SKA trading signals")
        
        signals = []
        
        # Process each correlation time window
        for _, corr_row in self.correlation_data.iterrows():
            timestamp = corr_row['timestamp']
            
            # Get entropy context for this time window
            entropy_context = self._get_entropy_context(timestamp)
            
            # Generate signal for this timestamp
            signal = self._generate_timestamp_signal(corr_row, entropy_context, timestamp)
            signals.append(signal)
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)
        self.signals = signals_df
        
        # Update current signal with latest
        if len(signals_df) > 0:
            self.current_signal = signals_df.iloc[-1].to_dict()
        
        # Calculate signal statistics
        self._calculate_signal_statistics()
        
        logger.info(f"Generated {len(signals_df)} trading signals")
        return signals_df
    
    def _get_entropy_context(self, timestamp: pd.Timestamp) -> Dict:
        """
        Get entropy analysis context for a specific timestamp.
        
        Args:
            timestamp (pd.Timestamp): Target timestamp
            
        Returns:
            Dict: Entropy context including structural probabilities
        """
        # Find entropy data around this timestamp (within reasonable window)
        time_window = pd.Timedelta(minutes=3)  # 3-minute context window
        
        # Convert timestamp to ensure timezone consistency
        target_time = pd.Timestamp(timestamp)
        if target_time.tz is not None:
            target_time = target_time.tz_localize(None)
        
        # Filter entropy data within time window
        entropy_timestamps = pd.to_datetime(self.entropy_data['timestamp'])
        if entropy_timestamps.dt.tz is not None:
            entropy_timestamps = entropy_timestamps.dt.tz_localize(None)
        
        window_mask = (
            (entropy_timestamps >= target_time - time_window) &
            (entropy_timestamps <= target_time + time_window)
        )
        
        window_data = self.entropy_data[window_mask]
        
        if len(window_data) == 0:
            # No entropy data in window, use global statistics
            return {
                'mean_structural_prob': float(self.entropy_data['structural_probability'].mean()),
                'mean_entropy': float(self.entropy_data['entropy'].mean()),
                'regime_distribution': dict(self.entropy_data['regime'].value_counts(normalize=True)),
                'sample_count': 0
            }
        
        # Calculate entropy context statistics
        context = {
            'mean_structural_prob': float(window_data['structural_probability'].mean()),
            'std_structural_prob': float(window_data['structural_probability'].std()),
            'mean_entropy': float(window_data['entropy'].mean()),
            'entropy_trend': float(window_data['entropy'].iloc[-1] - window_data['entropy'].iloc[0]) if len(window_data) > 1 else 0.0,
            'regime_distribution': dict(window_data['regime'].value_counts(normalize=True)),
            'dominant_regime': int(window_data['regime'].mode().iloc[0]) if len(window_data) > 0 else 0,
            'recent_transitions': len(window_data[window_data['regime'] != window_data['regime'].shift(1)].dropna()),
            'sample_count': len(window_data)
        }
        
        return context
    
    def _generate_timestamp_signal(self, corr_row: pd.Series, entropy_context: Dict, timestamp: pd.Timestamp) -> Dict:
        """
        Generate trading signal for specific timestamp combining both layers.
        
        Args:
            corr_row (pd.Series): Correlation analysis for this timestamp
            entropy_context (Dict): Entropy analysis context
            timestamp (pd.Timestamp): Signal timestamp
            
        Returns:
            Dict: Complete trading signal
        """
        # Extract correlation signals
        corr_up = corr_row['corr_up']
        corr_down = corr_row['corr_down']
        
        # Generate correlation-based signals
        corr_signal = self._interpret_correlation_signals(corr_up, corr_down)
        
        # Generate entropy-based signals  
        entropy_signal = self._interpret_entropy_signals(entropy_context)
        
        # Combine signals using weighted approach
        combined_signal = self._combine_signals(corr_signal, entropy_signal)
        
        # Create complete signal record
        signal = {
            'timestamp': timestamp,
            
            # Raw inputs
            'corr_up': corr_up,
            'corr_down': corr_down,
            'mean_structural_prob': entropy_context['mean_structural_prob'],
            'mean_entropy': entropy_context['mean_entropy'],
            
            # Layer-specific signals
            'correlation_signal': corr_signal['direction'].value,
            'correlation_strength': corr_signal['strength'].value,
            'correlation_confidence': corr_signal['confidence'],
            
            'entropy_signal': entropy_signal['direction'].value,
            'entropy_strength': entropy_signal['strength'].value,
            'entropy_confidence': entropy_signal['confidence'],
            
            # Combined signal
            'final_signal': combined_signal['direction'].value,
            'final_strength': combined_signal['strength'].value,
            'final_confidence': combined_signal['confidence'],
            'signal_score': combined_signal['score'],
            
            # Context
            'dominant_regime': entropy_context.get('dominant_regime', 0),
            'entropy_samples': entropy_context['sample_count'],
            'uptrend_samples': int(corr_row.get('uptrend_samples', 0)),
            'downtrend_samples': int(corr_row.get('downtrend_samples', 0))
        }
        
        return signal
    
    def _interpret_correlation_signals(self, corr_up: float, corr_down: float) -> Dict:
        """
        Interpret correlation values into trading signals.
        
        Args:
            corr_up (float): Neutral↔bull correlation
            corr_down (float): Neutral↔bear correlation
            
        Returns:
            Dict: Correlation signal interpretation
        """
        if np.isnan(corr_up) or np.isnan(corr_down):
            return {
                'direction': SignalDirection.NEUTRAL,
                'strength': SignalStrength.VERY_WEAK,
                'confidence': 0.0,
                'reasoning': 'Insufficient correlation data'
            }
        
        # Signal logic based on mathematical model thresholds
        if corr_up > 0.8 and corr_down < -0.3:
            return {
                'direction': SignalDirection.STRONG_BULL,
                'strength': SignalStrength.VERY_STRONG,
                'confidence': 0.95,
                'reasoning': 'Strong uptrend synchronization with downtrend anti-correlation'
            }
        elif corr_down > 0.8 and corr_up < -0.3:
            return {
                'direction': SignalDirection.STRONG_BEAR,
                'strength': SignalStrength.VERY_STRONG,
                'confidence': 0.95,
                'reasoning': 'Strong downtrend synchronization with uptrend anti-correlation'
            }
        elif corr_up > 0.3 and corr_down < 0.3:
            strength = SignalStrength.STRONG if corr_up > 0.6 else SignalStrength.MODERATE
            return {
                'direction': SignalDirection.MODERATE_BULL,
                'strength': strength,
                'confidence': 0.7,
                'reasoning': f'Moderate uptrend correlation ({corr_up:.3f})'
            }
        elif corr_down > 0.3 and corr_up < 0.3:
            strength = SignalStrength.STRONG if corr_down > 0.6 else SignalStrength.MODERATE
            return {
                'direction': SignalDirection.MODERATE_BEAR,
                'strength': strength,
                'confidence': 0.7,
                'reasoning': f'Moderate downtrend correlation ({corr_down:.3f})'
            }
        elif abs(corr_up) < 0.1 and abs(corr_down) < 0.1:
            return {
                'direction': SignalDirection.CONSOLIDATION,
                'strength': SignalStrength.MODERATE,
                'confidence': 0.6,
                'reasoning': 'Low correlation indicates consolidation'
            }
        elif corr_up < -0.5 or corr_down < -0.5:
            return {
                'direction': SignalDirection.REVERSAL_IMMINENT,
                'strength': SignalStrength.STRONG,
                'confidence': 0.8,
                'reasoning': 'Strong anti-correlation suggests reversal'
            }
        else:
            return {
                'direction': SignalDirection.NEUTRAL,
                'strength': SignalStrength.WEAK,
                'confidence': 0.3,
                'reasoning': 'Mixed correlation signals'
            }
    
    def _interpret_entropy_signals(self, entropy_context: Dict) -> Dict:
        """
        Interpret entropy analysis into trading signals.
        
        Args:
            entropy_context (Dict): Entropy analysis context
            
        Returns:
            Dict: Entropy signal interpretation
        """
        mean_prob = entropy_context['mean_structural_prob'] 
        mean_entropy = entropy_context['mean_entropy']
        dominant_regime = entropy_context.get('dominant_regime', 0)
        entropy_trend = entropy_context.get('entropy_trend', 0.0)
        
        # High structural probability = stable regime
        if mean_prob > 0.95:
            if dominant_regime == 1:  # Bull regime dominant
                return {
                    'direction': SignalDirection.MODERATE_BULL,
                    'strength': SignalStrength.STRONG,
                    'confidence': 0.85,
                    'reasoning': f'High structural probability ({mean_prob:.3f}) in bull regime'
                }
            elif dominant_regime == 2:  # Bear regime dominant
                return {
                    'direction': SignalDirection.MODERATE_BEAR,
                    'strength': SignalStrength.STRONG,
                    'confidence': 0.85,
                    'reasoning': f'High structural probability ({mean_prob:.3f}) in bear regime'
                }
            else:  # Neutral regime dominant
                return {
                    'direction': SignalDirection.CONSOLIDATION,
                    'strength': SignalStrength.MODERATE,
                    'confidence': 0.7,
                    'reasoning': f'High structural probability ({mean_prob:.3f}) in neutral regime'
                }
        
        # Low structural probability = regime instability
        elif mean_prob < 0.8:
            return {
                'direction': SignalDirection.REVERSAL_IMMINENT,
                'strength': SignalStrength.STRONG,
                'confidence': 0.8,
                'reasoning': f'Low structural probability ({mean_prob:.3f}) indicates regime instability'
            }
        
        # Entropy trend analysis
        elif abs(entropy_trend) > 0.1:
            if entropy_trend < 0:  # Entropy decreasing (information gain)
                return {
                    'direction': SignalDirection.WEAK_BULL if dominant_regime == 1 else SignalDirection.CONSOLIDATION,
                    'strength': SignalStrength.MODERATE,
                    'confidence': 0.6,
                    'reasoning': f'Information gain trend (entropy Δ={entropy_trend:.3f})'
                }
            else:  # Entropy increasing (information loss)
                return {
                    'direction': SignalDirection.WEAK_BEAR if dominant_regime == 2 else SignalDirection.NEUTRAL,
                    'strength': SignalStrength.MODERATE,
                    'confidence': 0.6,
                    'reasoning': f'Information loss trend (entropy Δ={entropy_trend:.3f})'
                }
        
        # Default: moderate confidence based on regime
        else:
            regime_directions = {
                0: SignalDirection.NEUTRAL,
                1: SignalDirection.WEAK_BULL,
                2: SignalDirection.WEAK_BEAR
            }
            
            return {
                'direction': regime_directions.get(dominant_regime, SignalDirection.NEUTRAL),
                'strength': SignalStrength.WEAK,
                'confidence': 0.4,
                'reasoning': f'Moderate entropy signals (P={mean_prob:.3f})'
            }
    
    def _combine_signals(self, corr_signal: Dict, entropy_signal: Dict) -> Dict:
        """
        Combine correlation and entropy signals using weighted approach.
        
        Args:
            corr_signal (Dict): Correlation signal
            entropy_signal (Dict): Entropy signal
            
        Returns:
            Dict: Combined signal
        """
        # Convert signal directions to numeric scores
        direction_scores = {
            SignalDirection.STRONG_BEAR: -2.0,
            SignalDirection.MODERATE_BEAR: -1.5,
            SignalDirection.WEAK_BEAR: -1.0,
            SignalDirection.REVERSAL_IMMINENT: -0.5,  # Depends on context
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.CONSOLIDATION: 0.0,
            SignalDirection.WEAK_BULL: 1.0,
            SignalDirection.MODERATE_BULL: 1.5,
            SignalDirection.STRONG_BULL: 2.0
        }
        
        # Convert strength to multiplier
        strength_multipliers = {
            SignalStrength.VERY_WEAK: 0.2,
            SignalStrength.WEAK: 0.4,
            SignalStrength.MODERATE: 0.6,
            SignalStrength.STRONG: 0.8,
            SignalStrength.VERY_STRONG: 1.0
        }
        
        # Calculate weighted scores
        corr_score = (
            direction_scores[corr_signal['direction']] * 
            strength_multipliers[corr_signal['strength']] * 
            corr_signal['confidence'] *
            self.correlation_weight
        )
        
        entropy_score = (
            direction_scores[entropy_signal['direction']] * 
            strength_multipliers[entropy_signal['strength']] * 
            entropy_signal['confidence'] *
            self.entropy_weight
        )
        
        # Combined score
        combined_score = corr_score + entropy_score
        
        # Combined confidence (weighted average)
        combined_confidence = (
            corr_signal['confidence'] * self.correlation_weight +
            entropy_signal['confidence'] * self.entropy_weight
        )
        
        # Determine final direction and strength
        if abs(combined_score) < self.signal_threshold:
            final_direction = SignalDirection.NEUTRAL
            final_strength = SignalStrength.WEAK
        elif combined_score > 1.5:
            final_direction = SignalDirection.STRONG_BULL
            final_strength = SignalStrength.VERY_STRONG
        elif combined_score > 1.0:
            final_direction = SignalDirection.MODERATE_BULL
            final_strength = SignalStrength.STRONG
        elif combined_score > 0.5:
            final_direction = SignalDirection.WEAK_BULL
            final_strength = SignalStrength.MODERATE
        elif combined_score < -1.5:
            final_direction = SignalDirection.STRONG_BEAR
            final_strength = SignalStrength.VERY_STRONG
        elif combined_score < -1.0:
            final_direction = SignalDirection.MODERATE_BEAR
            final_strength = SignalStrength.STRONG
        elif combined_score < -0.5:
            final_direction = SignalDirection.WEAK_BEAR
            final_strength = SignalStrength.MODERATE
        else:
            final_direction = SignalDirection.CONSOLIDATION
            final_strength = SignalStrength.MODERATE
        
        return {
            'direction': final_direction,
            'strength': final_strength,
            'confidence': combined_confidence,
            'score': combined_score,
            'components': {
                'correlation_score': corr_score,
                'entropy_score': entropy_score
            }
        }
    
    def _calculate_signal_statistics(self) -> None:
        """Calculate comprehensive signal statistics."""
        if len(self.signals) == 0:
            self.signal_statistics = {}
            return
        
        # Signal distribution analysis
        signal_counts = self.signals['final_signal'].value_counts()
        total_signals = len(self.signals)
        
        # Confidence analysis
        confidence_stats = {
            'mean_confidence': float(self.signals['final_confidence'].mean()),
            'std_confidence': float(self.signals['final_confidence'].std()),
            'high_confidence_signals': int((self.signals['final_confidence'] > 0.7).sum())
        }
        
        # Signal strength analysis
        strength_counts = self.signals['final_strength'].value_counts()
        
        # Score analysis
        score_stats = {
            'mean_score': float(self.signals['signal_score'].mean()),
            'std_score': float(self.signals['signal_score'].std()),
            'max_score': float(self.signals['signal_score'].max()),
            'min_score': float(self.signals['signal_score'].min())
        }
        
        self.signal_statistics = {
            'total_signals': total_signals,
            'signal_distribution': signal_counts.to_dict(),
            'signal_percentages': (signal_counts / total_signals * 100).to_dict(),
            'confidence_analysis': confidence_stats,
            'strength_distribution': strength_counts.to_dict(),
            'score_analysis': score_stats,
            'weights_used': {
                'correlation_weight': self.correlation_weight,
                'entropy_weight': self.entropy_weight
            }
        }
    
    def get_current_signal(self) -> Optional[Dict]:
        """Get the most recent trading signal."""
        return self.current_signal.copy() if self.current_signal else None
    
    def get_signal_statistics(self) -> Dict:
        """Get comprehensive signal statistics."""
        return self.signal_statistics.copy()
    
    def get_signal_summary(self) -> str:
        """
        Get formatted summary of signal analysis.
        
        Returns:
            str: Human-readable signal summary
        """
        if not self.signal_statistics:
            return "No signals generated"
        
        stats = self.signal_statistics
        current = self.current_signal
        
        # Format current signal safely
        if current:
            direction = current['final_signal'].upper()
            strength = current['final_strength'].upper()
            confidence = f"{current['final_confidence']:.3f}"
            score = f"{current['signal_score']:+.3f}"
        else:
            direction = strength = confidence = score = 'None'
        
        summary = f"""
SKA Dual-Layer Signal Analysis
==============================
Total Signals Generated: {stats['total_signals']}
Weighting: Correlation {self.correlation_weight:.1f} | Entropy {self.entropy_weight:.1f}

Current Signal:
  Direction: {direction}
  Strength: {strength}
  Confidence: {confidence}
  Score: {score}

Signal Distribution:"""
        
        for signal, count in stats['signal_distribution'].items():
            percentage = stats['signal_percentages'][signal]
            summary += f"\n  {signal}: {count} ({percentage:.1f}%)"
        
        summary += f"\n\nConfidence Analysis:"
        summary += f"\n  Mean Confidence: {stats['confidence_analysis']['mean_confidence']:.3f}"
        summary += f"\n  High Confidence Signals: {stats['confidence_analysis']['high_confidence_signals']}/{stats['total_signals']}"
        
        summary += f"\n\nScore Analysis:"
        summary += f"\n  Mean Score: {stats['score_analysis']['mean_score']:+.3f}"
        summary += f"\n  Range: {stats['score_analysis']['min_score']:+.3f} to {stats['score_analysis']['max_score']:+.3f}"
        
        return summary.strip()


def generate_ska_signals(correlation_df: pd.DataFrame, 
                        entropy_enhanced_data: pd.DataFrame,
                        correlation_weight: float = 0.6,
                        entropy_weight: float = 0.4) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to generate SKA trading signals.
    
    Args:
        correlation_df (pd.DataFrame): Correlation time series
        entropy_enhanced_data (pd.DataFrame): Data with structural probabilities
        correlation_weight (float): Weight for correlation signals
        entropy_weight (float): Weight for entropy signals
        
    Returns:
        Tuple[pd.DataFrame, Dict]: (signals_df, statistics)
    """
    generator = SKASignalGenerator(
        correlation_weight=correlation_weight,
        entropy_weight=entropy_weight
    )
    
    generator.initialize(correlation_df, entropy_enhanced_data)
    signals_df = generator.generate_signals()
    statistics = generator.get_signal_statistics()
    
    return signals_df, statistics


if __name__ == "__main__":
    # Test the signal generator with complete pipeline
    from data_loader import load_primary_data
    from regime_classifier import RegimeClassifier
    from transition_tracker import TransitionTracker
    from correlation_engine import CorrelationEngine
    from entropy_probability import EntropyProbabilityEngine
    
    file_path = "questdb-query-1751544843847.csv"
    
    try:
        # Load and process data
        data, metadata = load_primary_data(file_path)
        
        # Classify regimes
        classifier = RegimeClassifier()
        classified_data = classifier.classify_regimes(data)
        
        # Track transitions
        tracker = TransitionTracker()
        transition_data = tracker.track_transitions(classified_data)
        
        # Compute correlations
        corr_engine = CorrelationEngine(correlation_time_window=180.0)
        corr_engine.initialize_with_transitions(transition_data, classified_data)
        correlation_df = corr_engine.compute_correlations_over_time()
        
        # Compute entropy probabilities
        entropy_engine = EntropyProbabilityEngine()
        entropy_enhanced_data = entropy_engine.compute_structural_probabilities(classified_data)
        
        # Generate signals
        signal_generator = SKASignalGenerator(correlation_weight=0.6, entropy_weight=0.4)
        signal_generator.initialize(correlation_df, entropy_enhanced_data)
        signals_df = signal_generator.generate_signals()
        
        # Display results
        print(signal_generator.get_signal_summary())
        
        # Show sample signals
        if len(signals_df) > 0:
            print(f"\nSample Trading Signals:")
            display_cols = ['timestamp', 'final_signal', 'final_strength', 'final_confidence', 'signal_score']
            print(signals_df[display_cols].to_string(index=False))
        
        print(f"\n🚀 BREAKTHROUGH: First information-theoretic trading signals generated!")
        print(f"Dual-layer analysis combining correlation patterns with entropy physics!")
        
    except Exception as e:
        print(f"Error: {e}")