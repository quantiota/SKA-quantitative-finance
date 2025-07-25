"""
SKA Trading Algorithm - Transition Tracker Module

Δτ time series tracking for regime transitions in the SKA quantitative finance framework.
Builds the fundamental time-series data required for correlation analysis and signal generation.

Author: SKA Quantitative Finance Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransitionTracker:
    """
    Transition tracker for SKA trading algorithm.
    
    Tracks Δτ (delta tau) time series for all regime transitions as defined in the 
    mathematical model. Each transition type gets its own chronological time series
    of durations since the last distinct timestamp.
    
    Mathematical Foundation:
    - Δτᵢ→ⱼ(t) = time duration for transition from regime i to regime j at time t
    - For multiple trades at same timestamp: Δτ(tₖ) = tₖ - t_prev (shared value)
    - Builds {Δτᵢ→ⱼ(t)} series for correlation analysis
    """
    
    # Regime encoding (matching regime_classifier.py)
    REGIME_NEUTRAL = 0
    REGIME_BULL = 1  
    REGIME_BEAR = 2
    
    # All 9 possible transitions
    ALL_TRANSITIONS = [
        (REGIME_NEUTRAL, REGIME_NEUTRAL), (REGIME_NEUTRAL, REGIME_BULL), (REGIME_NEUTRAL, REGIME_BEAR),
        (REGIME_BULL, REGIME_NEUTRAL), (REGIME_BULL, REGIME_BULL), (REGIME_BULL, REGIME_BEAR),
        (REGIME_BEAR, REGIME_NEUTRAL), (REGIME_BEAR, REGIME_BULL), (REGIME_BEAR, REGIME_BEAR)
    ]
    
    # Trend pairs for correlation analysis (from mathematical model)
    TREND_UP_PAIRS = [(REGIME_NEUTRAL, REGIME_BULL), (REGIME_BULL, REGIME_NEUTRAL)]
    TREND_DOWN_PAIRS = [(REGIME_NEUTRAL, REGIME_BEAR), (REGIME_BEAR, REGIME_NEUTRAL)]
    
    def __init__(self):
        """Initialize transition tracker."""
        self.data = None
        self.transition_series = {}  # {(i,j): [δτ values]}
        self.transition_timestamps = {}  # {(i,j): [timestamps]}
        self.transition_indices = {}  # {(i,j): [data indices]}
        self.statistics = {}
        self.last_distinct_timestamp = None
        
    def track_transitions(self, classified_data: pd.DataFrame) -> Dict:
        """
        Build Δτ time series for all regime transitions.
        
        Args:
            classified_data (pd.DataFrame): Data with regime classification
            
        Returns:
            Dict: Complete transition tracking results
        """
        logger.info(f"Tracking transitions for {len(classified_data)} trades")
        
        # Store data
        self.data = classified_data.copy()
        
        # Initialize tracking structures
        self._initialize_tracking()
        
        # Process each trade to build time series
        self._process_transitions()
        
        # Calculate statistics
        self._calculate_statistics()
        
        logger.info(f"Transition tracking completed - {len(self.transition_series)} transition types found")
        return self.get_tracking_results()
    
    def _initialize_tracking(self) -> None:
        """Initialize all tracking data structures."""
        # Initialize time series for all possible transitions
        for transition in self.ALL_TRANSITIONS:
            self.transition_series[transition] = []
            self.transition_timestamps[transition] = []
            self.transition_indices[transition] = []
        
        logger.info("Tracking structures initialized for all 9 transition types")
    
    def _process_transitions(self) -> None:
        """Process each trade to build Δτ time series."""
        prev_timestamp = None
        
        for idx, row in self.data.iterrows():
            current_timestamp = row['timestamp']
            
            # Skip first trade (no previous regime)
            if idx == 0:
                prev_timestamp = current_timestamp
                self.last_distinct_timestamp = current_timestamp
                continue
            
            # Get transition type
            prev_regime = int(row['prev_regime']) if not pd.isna(row['prev_regime']) else None
            current_regime = int(row['regime'])
            
            if prev_regime is None:
                continue
            
            transition = (prev_regime, current_regime)
            
            # Calculate Δτ based on mathematical model
            delta_tau = self._calculate_delta_tau(current_timestamp, prev_timestamp)
            
            # Add to time series
            self.transition_series[transition].append(delta_tau)
            self.transition_timestamps[transition].append(current_timestamp)
            self.transition_indices[transition].append(idx)
            
            # Update tracking variables
            if current_timestamp != prev_timestamp:
                self.last_distinct_timestamp = current_timestamp
            prev_timestamp = current_timestamp
        
        logger.info(f"Processed {sum(len(series) for series in self.transition_series.values())} transitions")
    
    def _calculate_delta_tau(self, current_timestamp: datetime, prev_timestamp: datetime) -> float:
        """
        Calculate Δτ duration according to SKA mathematical model.
        
        Args:
            current_timestamp: Current trade timestamp
            prev_timestamp: Previous trade timestamp
            
        Returns:
            float: Δτ in seconds
        """
        # For trades at same timestamp: use time since last distinct timestamp
        if current_timestamp == prev_timestamp and self.last_distinct_timestamp is not None:
            delta_tau = (current_timestamp - self.last_distinct_timestamp).total_seconds()
        else:
            # For trades at different timestamps: use direct time difference
            delta_tau = (current_timestamp - prev_timestamp).total_seconds()
            
        return max(delta_tau, 0.001)  # Minimum 1ms to avoid division by zero
    
    def _calculate_statistics(self) -> None:
        """Calculate comprehensive transition statistics."""
        stats = {
            'transition_counts': {},
            'transition_durations': {},
            'trend_pair_analysis': {},
            'temporal_analysis': {},
            'summary': {}
        }
        
        total_transitions = 0
        
        # Analyze each transition type
        for transition, series in self.transition_series.items():
            if len(series) == 0:
                continue
                
            i, j = transition
            count = len(series)
            total_transitions += count
            
            # Basic statistics
            series_array = np.array(series)
            stats['transition_counts'][f"{i}→{j}"] = count
            stats['transition_durations'][f"{i}→{j}"] = {
                'count': count,
                'mean': float(np.mean(series_array)),
                'std': float(np.std(series_array)),
                'min': float(np.min(series_array)),
                'max': float(np.max(series_array)),
                'median': float(np.median(series_array))
            }
        
        # Trend pair analysis
        for pair_name, pairs in [('uptrend', self.TREND_UP_PAIRS), ('downtrend', self.TREND_DOWN_PAIRS)]:
            pair_data = {}
            for transition in pairs:
                if transition in self.transition_series and len(self.transition_series[transition]) > 0:
                    i, j = transition
                    pair_data[f"{i}→{j}"] = {
                        'count': len(self.transition_series[transition]),
                        'series_length': len(self.transition_series[transition]),
                        'mean_duration': float(np.mean(self.transition_series[transition]))
                    }
            stats['trend_pair_analysis'][pair_name] = pair_data
        
        # Temporal analysis
        if self.data is not None:
            time_span = (self.data['timestamp'].max() - self.data['timestamp'].min()).total_seconds()
            stats['temporal_analysis'] = {
                'total_time_span_seconds': float(time_span),
                'average_transition_rate': float(total_transitions / time_span) if time_span > 0 else 0.0,
                'total_transitions': total_transitions
            }
        
        # Summary
        dominant_transitions = sorted(stats['transition_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
        stats['summary'] = {
            'total_transition_types': len([t for t in self.transition_series if len(self.transition_series[t]) > 0]),
            'most_common_transitions': dominant_transitions,
            'trend_pairs_available': {
                'uptrend_pairs': len(stats['trend_pair_analysis'].get('uptrend', {})),
                'downtrend_pairs': len(stats['trend_pair_analysis'].get('downtrend', {}))
            }
        }
        
        self.statistics = stats
        logger.info(f"Statistics calculated for {stats['summary']['total_transition_types']} transition types")
    
    def get_transition_series(self, from_regime: int, to_regime: int) -> List[float]:
        """
        Get Δτ time series for specific transition.
        
        Args:
            from_regime (int): Source regime (0=neutral, 1=bull, 2=bear)
            to_regime (int): Target regime (0=neutral, 1=bull, 2=bear)
            
        Returns:
            List[float]: Δτ time series in chronological order
        """
        transition = (from_regime, to_regime)
        return self.transition_series.get(transition, []).copy()
    
    def get_trend_pair_series(self, trend_type: str) -> Dict[str, List[float]]:
        """
        Get Δτ time series for trend pairs (for correlation analysis).
        
        Args:
            trend_type (str): 'uptrend' or 'downtrend'
            
        Returns:
            Dict[str, List[float]]: Series for each transition in the trend pair
        """
        if trend_type == 'uptrend':
            pairs = self.TREND_UP_PAIRS
        elif trend_type == 'downtrend':
            pairs = self.TREND_DOWN_PAIRS
        else:
            raise ValueError(f"Invalid trend_type: {trend_type}. Must be 'uptrend' or 'downtrend'")
        
        result = {}
        for transition in pairs:
            i, j = transition
            key = f"{i}→{j}"
            result[key] = self.get_transition_series(i, j)
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get comprehensive transition tracking statistics."""
        return self.statistics.copy()
    
    def get_tracking_results(self) -> Dict:
        """
        Get complete tracking results for correlation analysis.
        
        Returns:
            Dict: All tracking data and statistics
        """
        return {
            'transition_series': {f"{i}→{j}": series.copy() for (i,j), series in self.transition_series.items()},
            'transition_timestamps': {f"{i}→{j}": ts.copy() for (i,j), ts in self.transition_timestamps.items()},
            'statistics': self.get_statistics(),
            'trend_pairs': {
                'uptrend': self.get_trend_pair_series('uptrend'),
                'downtrend': self.get_trend_pair_series('downtrend')
            }
        }
    
    def get_tracking_summary(self) -> str:
        """
        Get formatted summary of transition tracking.
        
        Returns:
            str: Human-readable tracking summary
        """
        if not self.statistics:
            return "No transition tracking performed"
        
        stats = self.statistics
        summary_stats = stats.get('summary', {})
        temporal_stats = stats.get('temporal_analysis', {})
        
        summary = f"""
SKA Transition Tracking Summary
===============================
Total Transition Types: {summary_stats.get('total_transition_types', 0)}
Total Transitions: {temporal_stats.get('total_transitions', 0):,}
Time Span: {temporal_stats.get('total_time_span_seconds', 0):.1f} seconds
Transition Rate: {temporal_stats.get('average_transition_rate', 0):.2f} per second

Most Common Transitions:"""
        
        for transition, count in summary_stats.get('most_common_transitions', []):
            percentage = (count / temporal_stats.get('total_transitions', 1)) * 100
            summary += f"\n  {transition}: {count:,} ({percentage:.1f}%)"
        
        # Add trend pair availability
        trend_pairs = summary_stats.get('trend_pairs_available', {})
        uptrend_pairs = stats.get('trend_pair_analysis', {}).get('uptrend', {})
        downtrend_pairs = stats.get('trend_pair_analysis', {}).get('downtrend', {})
        
        summary += f"\n\nTrend Pair Analysis:"
        summary += f"\n  Uptrend pairs available: {len(uptrend_pairs)}"
        for pair, data in uptrend_pairs.items():
            summary += f"\n    {pair}: {data['count']} transitions (avg: {data['mean_duration']:.3f}s)"
        
        summary += f"\n  Downtrend pairs available: {len(downtrend_pairs)}"
        for pair, data in downtrend_pairs.items():
            summary += f"\n    {pair}: {data['count']} transitions (avg: {data['mean_duration']:.3f}s)"
        
        return summary.strip()
    
    def validate_for_correlation_analysis(self) -> Tuple[bool, List[str]]:
        """
        Validate transition tracking for correlation analysis requirements.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        if not self.statistics:
            return False, ["No transition tracking performed"]
        
        # Check trend pair availability
        uptrend_series = self.get_trend_pair_series('uptrend')
        downtrend_series = self.get_trend_pair_series('downtrend')
        
        min_transitions = 10
        
        # Check uptrend pairs
        uptrend_counts = [len(series) for series in uptrend_series.values()]
        if not uptrend_counts or min(uptrend_counts) < min_transitions:
            issues.append(f"Insufficient uptrend transitions: {uptrend_counts} (minimum {min_transitions} each)")
        
        # Check downtrend pairs
        downtrend_counts = [len(series) for series in downtrend_series.values()]
        if not downtrend_counts or min(downtrend_counts) < min_transitions:
            issues.append(f"Insufficient downtrend transitions: {downtrend_counts} (minimum {min_transitions} each)")
        
        # Check for time series variation
        for trend_type, series_dict in [('uptrend', uptrend_series), ('downtrend', downtrend_series)]:
            for pair_name, series in series_dict.items():
                if len(series) > 1 and np.std(series) < 0.001:
                    issues.append(f"No time variation in {trend_type} {pair_name}: std={np.std(series):.6f}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Transition tracking validation for correlation analysis: PASSED")
        else:
            logger.warning(f"⚠ Transition tracking validation issues: {len(issues)} issues")
        
        return is_valid, issues


def track_regime_transitions(classified_data: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Convenience function to track regime transitions.
    
    Args:
        classified_data (pd.DataFrame): Data with regime classification
        
    Returns:
        Tuple[Dict, Dict]: (tracking_results, statistics)
    """
    tracker = TransitionTracker()
    tracking_results = tracker.track_transitions(classified_data)
    statistics = tracker.get_statistics()
    
    # Validate for correlation analysis
    is_valid, issues = tracker.validate_for_correlation_analysis()
    if not is_valid:
        logger.warning("Transition tracking validation warnings:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return tracking_results, statistics


if __name__ == "__main__":
    # Test the transition tracker with classified data
    from data_loader import load_primary_data
    from regime_classifier import RegimeClassifier
    
    file_path = "questdb-query-1751544843847.csv"
    
    try:
        # Load and classify data
        data, metadata = load_primary_data(file_path)
        classifier = RegimeClassifier()
        classified_data = classifier.classify_regimes(data)
        
        # Track transitions
        tracker = TransitionTracker()
        tracking_results = tracker.track_transitions(classified_data)
        
        # Display summary
        print(tracker.get_tracking_summary())
        
        # Validate for correlation analysis
        is_valid, issues = tracker.validate_for_correlation_analysis()
        print(f"\nCorrelation Analysis Ready: {'✓ YES' if is_valid else '✗ NO'}")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Show sample trend pair data
        uptrend_series = tracker.get_trend_pair_series('uptrend')
        print(f"\nUptrend Pair Sample Data:")
        for pair, series in uptrend_series.items():
            if len(series) > 0:
                print(f"  {pair}: {len(series)} transitions, first 5 Δτ values: {series[:5]}")
                
    except Exception as e:
        print(f"Error: {e}")