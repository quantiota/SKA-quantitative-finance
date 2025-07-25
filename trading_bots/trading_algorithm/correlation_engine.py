"""
SKA Trading Algorithm - Correlation Engine Module

Time-windowed Pearson correlation computation for trend pairs in the SKA quantitative finance framework.
Implements the core correlation analysis that drives trading signal generation.

Author: SKA Quantitative Finance Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrelationEngine:
    """
    Correlation engine for SKA trading algorithm.
    
    Computes time-windowed Pearson correlations between paired regime transitions
    as defined in the mathematical model:
    
    ρ_up = PearsonCorr({Δτ₀→₁(t)}, {Δτ₁→₀(t)})    # neutral↔bull correlation
    ρ_down = PearsonCorr({Δτ₀→₂(t)}, {Δτ₂→₀(t)})  # neutral↔bear correlation
    
    Uses sliding time windows for real-time correlation tracking and signal generation.
    """
    
    # Regime encoding
    REGIME_NEUTRAL = 0
    REGIME_BULL = 1
    REGIME_BEAR = 2
    
    # Trend pair definitions
    UPTREND_PAIRS = {
        'neutral_to_bull': (REGIME_NEUTRAL, REGIME_BULL),
        'bull_to_neutral': (REGIME_BULL, REGIME_NEUTRAL)
    }
    
    DOWNTREND_PAIRS = {
        'neutral_to_bear': (REGIME_NEUTRAL, REGIME_BEAR),
        'bear_to_neutral': (REGIME_BEAR, REGIME_NEUTRAL)  
    }
    
    # Default time windows (from mathematical model)
    DEFAULT_WINDOWS = {
        'micro': 30,      # 30 seconds - immediate patterns
        'short': 60,      # 1 minute - standard window
        'medium': 120,    # 2 minutes - balanced sensitivity  
        'balanced': 180,  # 3 minutes - recommended default
        'trend': 300,     # 5 minutes - trend analysis
        'stable': 600     # 10 minutes - stable patterns
    }
    
    def __init__(self, correlation_time_window: float = 180.0, min_samples: int = 5):
        """
        Initialize correlation engine.
        
        Args:
            correlation_time_window (float): Time window in seconds for correlation computation
            min_samples (int): Minimum samples required for reliable correlation
        """
        self.correlation_time_window = correlation_time_window
        self.min_samples = min_samples
        
        # Data storage
        self.transition_data = None
        self.classified_data = None
        
        # Real-time correlation tracking
        self.correlation_history = {
            'uptrend': deque(),
            'downtrend': deque(),
            'timestamps': deque()
        }
        
        # Current correlations
        self.current_correlations = {
            'corr_up': np.nan,
            'corr_down': np.nan,
            'timestamp': None,
            'window_size': self.correlation_time_window,
            'samples_used': {'uptrend': 0, 'downtrend': 0}
        }
        
        # Statistics
        self.statistics = {}
        
    def initialize_with_transitions(self, transition_data: Dict, classified_data: pd.DataFrame) -> None:
        """
        Initialize engine with transition tracking data.
        
        Args:
            transition_data (Dict): Results from TransitionTracker
            classified_data (pd.DataFrame): Classified trading data with timestamps
        """
        self.transition_data = transition_data
        self.classified_data = classified_data.copy()
        
        logger.info(f"Correlation engine initialized with {len(classified_data)} trades")
        logger.info(f"Time window: {self.correlation_time_window}s, Min samples: {self.min_samples}")
    
    def compute_correlations_over_time(self) -> pd.DataFrame:
        """
        Compute correlations over sliding time windows for entire dataset.
        
        Returns:
            pd.DataFrame: Correlation time series with timestamps
        """
        if self.transition_data is None:
            raise ValueError("Must initialize with transition data first")
        
        logger.info("Computing correlations over sliding time windows")
        
        # Get timestamps for windowing
        timestamps = self.classified_data['timestamp'].values
        start_time = timestamps[0]
        end_time = timestamps[-1]
        
        # Create time series results
        correlation_series = []
        
        # Convert numpy datetime64 to pandas Timestamp for arithmetic
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        
        # Slide window across entire time range
        current_time = start_time + pd.Timedelta(seconds=self.correlation_time_window)
        
        while current_time <= end_time:
            # Compute correlation for current window
            corr_up, corr_down, samples = self._compute_window_correlation(current_time)
            
            correlation_series.append({
                'timestamp': current_time,
                'corr_up': corr_up,
                'corr_down': corr_down,
                'uptrend_samples': samples['uptrend'],
                'downtrend_samples': samples['downtrend'],
                'window_seconds': self.correlation_time_window
            })
            
            # Move window forward (smaller steps for more resolution)
            current_time += pd.Timedelta(seconds=self.correlation_time_window / 10)
        
        correlation_df = pd.DataFrame(correlation_series)
        
        # Update current correlations with latest values
        if len(correlation_df) > 0:
            latest = correlation_df.iloc[-1]
            self.current_correlations.update({
                'corr_up': latest['corr_up'],
                'corr_down': latest['corr_down'], 
                'timestamp': latest['timestamp'],
                'samples_used': {
                    'uptrend': int(latest['uptrend_samples']),
                    'downtrend': int(latest['downtrend_samples'])
                }
            })
        
        logger.info(f"Computed correlations for {len(correlation_df)} time windows")
        return correlation_df
    
    def _compute_window_correlation(self, window_end_time: datetime) -> Tuple[float, float, Dict]:
        """
        Compute correlation for a specific time window.
        
        Args:
            window_end_time (datetime): End time of the correlation window
            
        Returns:
            Tuple[float, float, Dict]: (corr_up, corr_down, sample_counts)
        """
        window_start_time = window_end_time - pd.Timedelta(seconds=self.correlation_time_window)
        
        # Filter transitions within time window
        uptrend_series = self._get_windowed_series('uptrend', window_start_time, window_end_time)
        downtrend_series = self._get_windowed_series('downtrend', window_start_time, window_end_time)
        
        # Compute correlations
        corr_up = self._compute_trend_correlation(uptrend_series)
        corr_down = self._compute_trend_correlation(downtrend_series)
        
        sample_counts = {
            'uptrend': min(len(series) for series in uptrend_series.values()) if uptrend_series else 0,
            'downtrend': min(len(series) for series in downtrend_series.values()) if downtrend_series else 0
        }
        
        return corr_up, corr_down, sample_counts
    
    def _get_windowed_series(self, trend_type: str, start_time: datetime, end_time: datetime) -> Dict[str, List[float]]:
        """
        Get transition series within time window for trend type.
        
        Args:
            trend_type (str): 'uptrend' or 'downtrend'
            start_time (datetime): Window start time
            end_time (datetime): Window end time
            
        Returns:
            Dict[str, List[float]]: Filtered Δτ series for trend pairs
        """
        if trend_type == 'uptrend':
            pairs = self.UPTREND_PAIRS
        elif trend_type == 'downtrend':
            pairs = self.DOWNTREND_PAIRS
        else:
            raise ValueError(f"Invalid trend_type: {trend_type}")
        
        windowed_series = {}
        
        for pair_name, (from_regime, to_regime) in pairs.items():
            transition_key = f"{from_regime}→{to_regime}"
            
            # Get full series and timestamps
            full_series = self.transition_data['transition_series'].get(transition_key, [])
            full_timestamps = self.transition_data['transition_timestamps'].get(transition_key, [])
            
            if len(full_series) == 0 or len(full_timestamps) == 0:
                windowed_series[pair_name] = []
                continue
            
            # Filter by time window (ensure timezone consistency)
            windowed_values = []
            for i, timestamp in enumerate(full_timestamps):
                # Convert timestamp to pandas Timestamp if needed
                ts = pd.Timestamp(timestamp)
                start_ts = pd.Timestamp(start_time)
                end_ts = pd.Timestamp(end_time)
                
                # Make timezone-naive for comparison if needed
                if ts.tz is not None:
                    ts = ts.tz_localize(None)
                if start_ts.tz is not None:
                    start_ts = start_ts.tz_localize(None)
                if end_ts.tz is not None:
                    end_ts = end_ts.tz_localize(None)
                
                if start_ts <= ts <= end_ts:
                    windowed_values.append(full_series[i])
            
            windowed_series[pair_name] = windowed_values
        
        return windowed_series
    
    def _compute_trend_correlation(self, trend_series: Dict[str, List[float]]) -> float:
        """
        Compute Pearson correlation for trend pair series.
        
        Args:
            trend_series (Dict[str, List[float]]): Δτ series for trend pair
            
        Returns:
            float: Pearson correlation coefficient (NaN if insufficient data)
        """
        if len(trend_series) != 2:
            return np.nan
        
        series_list = list(trend_series.values())
        series1, series2 = series_list[0], series_list[1]
        
        # Check minimum samples
        if len(series1) < self.min_samples or len(series2) < self.min_samples:
            return np.nan
        
        # Align series lengths (as per mathematical model section 7.2)
        min_length = min(len(series1), len(series2))
        if min_length < self.min_samples:
            return np.nan
        
        # Truncate to equal length (preserve most recent data)
        aligned_series1 = series1[-min_length:] if len(series1) > min_length else series1
        aligned_series2 = series2[-min_length:] if len(series2) > min_length else series2
        
        # Check for zero variance
        if np.std(aligned_series1) == 0 or np.std(aligned_series2) == 0:
            return 0.0
        
        # Compute Pearson correlation
        try:
            correlation, p_value = pearsonr(aligned_series1, aligned_series2)
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return np.nan
    
    def get_current_correlations(self) -> Dict:
        """
        Get current correlation values.
        
        Returns:
            Dict: Current correlation state
        """
        return self.current_correlations.copy()
    
    def get_correlation_signal(self, corr_up: Optional[float] = None, corr_down: Optional[float] = None) -> Dict:
        """
        Generate trading signal based on correlation thresholds.
        
        Args:
            corr_up (Optional[float]): Uptrend correlation (uses current if None)
            corr_down (Optional[float]): Downtrend correlation (uses current if None)
            
        Returns:
            Dict: Trading signal with interpretation
        """
        if corr_up is None:
            corr_up = self.current_correlations['corr_up']
        if corr_down is None:
            corr_down = self.current_correlations['corr_down']
        
        # Signal interpretation thresholds (from mathematical model)
        signal = {
            'corr_up': corr_up,
            'corr_down': corr_down,
            'timestamp': self.current_correlations['timestamp'],
            'uptrend_signal': self._interpret_correlation(corr_up),
            'downtrend_signal': self._interpret_correlation(corr_down),
            'combined_signal': 'neutral',
            'confidence': 'low'
        }
        
        # Combined signal logic
        if not np.isnan(corr_up) and not np.isnan(corr_down):
            if corr_up > 0.8 and corr_down < -0.3:
                signal['combined_signal'] = 'strong_bull'
                signal['confidence'] = 'high'
            elif corr_down > 0.8 and corr_up < -0.3:
                signal['combined_signal'] = 'strong_bear'
                signal['confidence'] = 'high'
            elif corr_up > 0.3 and corr_down < 0.3:
                signal['combined_signal'] = 'moderate_bull'
                signal['confidence'] = 'medium'
            elif corr_down > 0.3 and corr_up < 0.3:
                signal['combined_signal'] = 'moderate_bear'
                signal['confidence'] = 'medium'
            elif abs(corr_up) < 0.1 and abs(corr_down) < 0.1:
                signal['combined_signal'] = 'consolidation'
                signal['confidence'] = 'medium'
        
        return signal
    
    def _interpret_correlation(self, correlation: float) -> Dict:
        """
        Interpret correlation value according to SKA framework.
        
        Args:
            correlation (float): Correlation coefficient
            
        Returns:
            Dict: Signal interpretation
        """
        if np.isnan(correlation):
            return {'signal': 'insufficient_data', 'strength': 'none', 'description': 'Insufficient data'}
        
        if correlation >= 0.8:
            return {'signal': 'strong_sync', 'strength': 'high', 'description': 'Strong synchronized cycling'}
        elif correlation >= 0.3:
            return {'signal': 'moderate_sync', 'strength': 'medium', 'description': 'Moderate cycling pattern'}
        elif correlation >= -0.1:
            return {'signal': 'weak_random', 'strength': 'low', 'description': 'Weak/random transitions'}
        elif correlation >= -0.3:
            return {'signal': 'slight_anti', 'strength': 'low', 'description': 'Slight anti-correlation'}
        else:
            return {'signal': 'strong_anti', 'strength': 'high', 'description': 'Strong anti-correlation'}
    
    def analyze_correlation_dynamics(self, correlation_df: pd.DataFrame) -> Dict:
        """
        Analyze correlation dynamics over time.
        
        Args:
            correlation_df (pd.DataFrame): Correlation time series
            
        Returns:
            Dict: Comprehensive correlation analysis
        """
        if len(correlation_df) == 0:
            return {}
        
        # Filter valid correlations
        valid_up = correlation_df['corr_up'].dropna()
        valid_down = correlation_df['corr_down'].dropna()
        
        analysis = {
            'uptrend_correlation': {
                'mean': float(valid_up.mean()) if len(valid_up) > 0 else np.nan,
                'std': float(valid_up.std()) if len(valid_up) > 0 else np.nan,
                'min': float(valid_up.min()) if len(valid_up) > 0 else np.nan,
                'max': float(valid_up.max()) if len(valid_up) > 0 else np.nan,
                'valid_periods': int(len(valid_up))
            },
            'downtrend_correlation': {
                'mean': float(valid_down.mean()) if len(valid_down) > 0 else np.nan,
                'std': float(valid_down.std()) if len(valid_down) > 0 else np.nan,
                'min': float(valid_down.min()) if len(valid_down) > 0 else np.nan,
                'max': float(valid_down.max()) if len(valid_down) > 0 else np.nan,
                'valid_periods': int(len(valid_down))
            },
            'signal_distribution': self._analyze_signal_distribution(correlation_df),
            'temporal_stability': self._analyze_temporal_stability(correlation_df)
        }
        
        self.statistics = analysis
        return analysis
    
    def _analyze_signal_distribution(self, correlation_df: pd.DataFrame) -> Dict:
        """Analyze distribution of correlation signals."""
        signals = []
        for _, row in correlation_df.iterrows():
            signal = self.get_correlation_signal(row['corr_up'], row['corr_down'])
            signals.append(signal['combined_signal'])
        
        if not signals:
            return {}
        
        signal_counts = pd.Series(signals).value_counts()
        total = len(signals)
        
        return {
            'total_signals': total,
            'signal_counts': signal_counts.to_dict(),
            'signal_percentages': (signal_counts / total * 100).to_dict()
        }
    
    def _analyze_temporal_stability(self, correlation_df: pd.DataFrame) -> Dict:
        """Analyze temporal stability of correlations."""
        if len(correlation_df) < 2:
            return {}
        
        # Correlation changes between periods
        up_changes = correlation_df['corr_up'].diff().abs()
        down_changes = correlation_df['corr_down'].diff().abs()
        
        return {
            'uptrend_volatility': float(up_changes.mean()) if not up_changes.isna().all() else np.nan,
            'downtrend_volatility': float(down_changes.mean()) if not down_changes.isna().all() else np.nan,
            'max_uptrend_change': float(up_changes.max()) if not up_changes.isna().all() else np.nan,
            'max_downtrend_change': float(down_changes.max()) if not down_changes.isna().all() else np.nan
        }
    
    def get_correlation_summary(self) -> str:
        """
        Get formatted summary of correlation analysis.
        
        Returns:
            str: Human-readable correlation summary
        """
        current = self.current_correlations
        
        if np.isnan(current['corr_up']) and np.isnan(current['corr_down']):
            return "No correlation data available"
        
        summary = f"""
SKA Correlation Analysis Summary
================================
Time Window: {current['window_size']:.1f} seconds
Timestamp: {current['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if current['timestamp'] else 'N/A'}

Current Correlations:
  ρ_up (neutral↔bull):  {current['corr_up']:+.3f}
  ρ_down (neutral↔bear): {current['corr_down']:+.3f}

Sample Sizes:
  Uptrend samples: {current['samples_used']['uptrend']}
  Downtrend samples: {current['samples_used']['downtrend']}

Signal Interpretation:"""
        
        signal = self.get_correlation_signal()
        summary += f"\n  Combined Signal: {signal['combined_signal'].upper()}"
        summary += f"\n  Confidence: {signal['confidence'].upper()}"
        summary += f"\n  Uptrend: {signal['uptrend_signal']['description']}"
        summary += f"\n  Downtrend: {signal['downtrend_signal']['description']}"
        
        return summary.strip()


def compute_ska_correlations(transition_data: Dict, classified_data: pd.DataFrame, 
                           time_window: float = 180.0) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to compute SKA correlations.
    
    Args:
        transition_data (Dict): Transition tracking results
        classified_data (pd.DataFrame): Classified trading data
        time_window (float): Correlation time window in seconds
        
    Returns:
        Tuple[pd.DataFrame, Dict]: (correlation_time_series, analysis)
    """
    engine = CorrelationEngine(correlation_time_window=time_window)
    engine.initialize_with_transitions(transition_data, classified_data)
    
    correlation_df = engine.compute_correlations_over_time()
    analysis = engine.analyze_correlation_dynamics(correlation_df)
    
    return correlation_df, analysis


if __name__ == "__main__":
    # Test the correlation engine with full pipeline
    from data_loader import load_primary_data
    from regime_classifier import RegimeClassifier
    from transition_tracker import TransitionTracker
    
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
        engine = CorrelationEngine(correlation_time_window=180.0)
        engine.initialize_with_transitions(transition_data, classified_data)
        
        correlation_df = engine.compute_correlations_over_time()
        analysis = engine.analyze_correlation_dynamics(correlation_df)
        
        # Display results
        print(engine.get_correlation_summary())
        
        print(f"\nCorrelation Time Series:")
        print(f"Total periods: {len(correlation_df)}")
        if len(correlation_df) > 0:
            print(f"Valid uptrend correlations: {correlation_df['corr_up'].notna().sum()}")
            print(f"Valid downtrend correlations: {correlation_df['corr_down'].notna().sum()}")
            
            # Show sample periods
            valid_periods = correlation_df.dropna(subset=['corr_up', 'corr_down'])
            if len(valid_periods) > 0:
                print(f"\nSample Correlation Periods:")
                display_cols = ['timestamp', 'corr_up', 'corr_down', 'uptrend_samples', 'downtrend_samples']
                print(valid_periods[display_cols].head().to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")