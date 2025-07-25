"""
SKA Trading Algorithm - Regime Classifier Module

Bull/bear/neutral regime classification from price returns for the SKA quantitative finance framework.
Implements the 3-regime classification system required for correlation analysis and signal generation.

Author: SKA Quantitative Finance Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Regime classifier for SKA trading algorithm.
    
    Classifies each trade into one of three market regimes based on price returns:
    - Bull (1): Price increased from previous trade (return > 0)
    - Neutral (0): Price unchanged from previous trade (return = 0)  
    - Bear (2): Price decreased from previous trade (return < 0)
    
    This classification is fundamental to the SKA framework's correlation analysis.
    """
    
    # Regime encoding
    REGIME_BULL = 1
    REGIME_NEUTRAL = 0
    REGIME_BEAR = 2
    
    # Regime names for display
    REGIME_NAMES = {
        REGIME_BULL: 'bull',
        REGIME_NEUTRAL: 'neutral', 
        REGIME_BEAR: 'bear'
    }
    
    # Trend pairs for correlation analysis (from mathematical model)
    TREND_UP_PAIRS = [(REGIME_NEUTRAL, REGIME_BULL), (REGIME_BULL, REGIME_NEUTRAL)]
    TREND_DOWN_PAIRS = [(REGIME_NEUTRAL, REGIME_BEAR), (REGIME_BEAR, REGIME_NEUTRAL)]
    
    def __init__(self, price_precision: float = 1e-6):
        """
        Initialize regime classifier.
        
        Args:
            price_precision (float): Minimum price change to consider non-neutral
        """
        self.price_precision = price_precision
        self.data = None
        self.regimes = None
        self.transitions = None
        self.statistics = {}
        
    def classify_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all trades into bull/bear/neutral regimes.
        
        Args:
            data (pd.DataFrame): Trading data with 'price' column
            
        Returns:
            pd.DataFrame: Data with added regime classification columns
        """
        logger.info(f"Classifying regimes for {len(data)} trades")
        
        # Store data
        self.data = data.copy()
        
        # Calculate price returns
        self._calculate_price_returns()
        
        # Classify regimes based on returns
        self._classify_by_returns()
        
        # Detect regime transitions
        self._detect_transitions()
        
        # Calculate statistics
        self._calculate_statistics()
        
        logger.info("Regime classification completed")
        return self.data
    
    def _calculate_price_returns(self) -> None:
        """Calculate price returns between consecutive trades."""
        # Calculate absolute price changes
        self.data['price_change'] = self.data['price'].diff()
        
        # Calculate percentage returns
        self.data['price_return'] = self.data['price'].pct_change() * 100
        
        # Fill NaN for first trade (no previous price to compare)
        self.data.loc[0, 'price_change'] = 0.0
        self.data.loc[0, 'price_return'] = 0.0
        
        logger.info(f"Price returns calculated - Range: {self.data['price_return'].min():.6f}% to {self.data['price_return'].max():.6f}%")
    
    def _classify_by_returns(self) -> None:
        """Classify regimes based on price returns."""
        def classify_return(price_change: float) -> int:
            """Classify individual price change into regime."""
            if abs(price_change) < self.price_precision:
                return self.REGIME_NEUTRAL
            elif price_change > 0:
                return self.REGIME_BULL
            else:
                return self.REGIME_BEAR
        
        # Apply classification
        self.data['regime'] = self.data['price_change'].apply(classify_return)
        
        # Add regime names for readability
        self.data['regime_name'] = self.data['regime'].map(self.REGIME_NAMES)
        
        # Store regimes array for quick access
        self.regimes = self.data['regime'].values
        
        logger.info("Regime classification by returns completed")
    
    def _detect_transitions(self) -> None:
        """Detect regime transitions between consecutive trades."""
        # Calculate previous regime
        self.data['prev_regime'] = self.data['regime'].shift(1)
        
        # Create transition labels
        self.data['transition'] = (
            self.data['prev_regime'].map(self.REGIME_NAMES).fillna('start') + 
            '→' + 
            self.data['regime_name']
        )
        
        # Filter out the first trade (no previous regime)
        transition_mask = ~self.data['prev_regime'].isna()
        self.transitions = self.data[transition_mask].copy()
        
        logger.info(f"Detected {len(self.transitions)} regime transitions")
    
    def _calculate_statistics(self) -> None:
        """Calculate comprehensive regime and transition statistics."""
        # Regime distribution
        regime_counts = self.data['regime_name'].value_counts()
        total_trades = len(self.data)
        
        regime_stats = {}
        for regime_name in ['bull', 'neutral', 'bear']:
            count = regime_counts.get(regime_name, 0)
            percentage = (count / total_trades) * 100
            regime_stats[regime_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        # Transition distribution
        if len(self.transitions) > 0:
            transition_counts = self.transitions['transition'].value_counts()
            total_transitions = len(self.transitions)
            
            transition_stats = {}
            for transition, count in transition_counts.items():
                percentage = (count / total_transitions) * 100
                transition_stats[transition] = {
                    'count': int(count),
                    'percentage': float(percentage)
                }
        else:
            transition_stats = {}
        
        # Price statistics by regime
        price_by_regime = {}
        for regime_name in ['bull', 'neutral', 'bear']:
            regime_data = self.data[self.data['regime_name'] == regime_name]
            if len(regime_data) > 0:
                price_by_regime[regime_name] = {
                    'mean_price': float(regime_data['price'].mean()),
                    'mean_return': float(regime_data['price_return'].mean()),
                    'std_return': float(regime_data['price_return'].std()),
                }
            else:
                price_by_regime[regime_name] = {
                    'mean_price': 0.0,
                    'mean_return': 0.0,
                    'std_return': 0.0,
                }
        
        # Store all statistics
        self.statistics = {
            'regime_distribution': regime_stats,
            'transition_distribution': transition_stats,
            'price_by_regime': price_by_regime,
            'total_trades': int(total_trades),
            'total_transitions': int(len(self.transitions)) if self.transitions is not None else 0,
            'classification_summary': {
                'dominant_regime': regime_counts.index[0] if len(regime_counts) > 0 else 'unknown',
                'regime_diversity': float(len(regime_counts)),
                'neutral_dominance': float(regime_stats.get('neutral', {}).get('percentage', 0.0))
            }
        }
        
        logger.info(f"Regime statistics calculated - Dominant: {self.statistics['classification_summary']['dominant_regime']}")
    
    def get_regime_sequence(self) -> np.ndarray:
        """
        Get regime sequence as numpy array.
        
        Returns:
            np.ndarray: Array of regime codes (0=neutral, 1=bull, 2=bear)
        """
        if self.regimes is None:
            raise ValueError("No regimes classified yet. Call classify_regimes() first.")
        return self.regimes.copy()
    
    def get_transitions_data(self) -> pd.DataFrame:
        """
        Get transition data for correlation analysis.
        
        Returns:
            pd.DataFrame: Transitions with timing information
        """
        if self.transitions is None:
            raise ValueError("No transitions detected yet. Call classify_regimes() first.")
        return self.transitions.copy()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive regime classification statistics."""
        return self.statistics.copy()
    
    def get_regime_summary(self) -> str:
        """
        Get formatted summary of regime classification.
        
        Returns:
            str: Human-readable classification summary
        """
        if not self.statistics:
            return "No regime classification performed"
        
        stats = self.statistics
        regime_dist = stats['regime_distribution']
        
        summary = f"""
SKA Regime Classification Summary
=================================
Total Trades: {stats['total_trades']:,}
Total Transitions: {stats['total_transitions']:,}
Dominant Regime: {stats['classification_summary']['dominant_regime'].title()}

Regime Distribution:
  Bull:    {regime_dist.get('bull', {}).get('count', 0):4,} trades ({regime_dist.get('bull', {}).get('percentage', 0.0):5.1f}%)
  Neutral: {regime_dist.get('neutral', {}).get('count', 0):4,} trades ({regime_dist.get('neutral', {}).get('percentage', 0.0):5.1f}%)
  Bear:    {regime_dist.get('bear', {}).get('count', 0):4,} trades ({regime_dist.get('bear', {}).get('percentage', 0.0):5.1f}%)

Price Statistics by Regime:
  Bull - Mean Return: {stats['price_by_regime']['bull']['mean_return']:+.6f}%
  Neutral - Mean Return: {stats['price_by_regime']['neutral']['mean_return']:+.6f}%
  Bear - Mean Return: {stats['price_by_regime']['bear']['mean_return']:+.6f}%

Top Transitions:"""
        
        # Add top 5 transitions
        transition_dist = stats.get('transition_distribution', {})
        sorted_transitions = sorted(transition_dist.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for i, (transition, data) in enumerate(sorted_transitions[:5]):
            summary += f"\n  {transition}: {data['count']:3,} ({data['percentage']:5.1f}%)"
        
        return summary.strip()
    
    def validate_for_correlation_analysis(self) -> Tuple[bool, List[str]]:
        """
        Validate regime classification for correlation analysis requirements.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        if self.data is None or self.regimes is None:
            return False, ["No regime classification performed"]
        
        stats = self.statistics
        regime_dist = stats.get('regime_distribution', {})
        
        # Check for sufficient data in each regime
        min_regime_trades = 10
        for regime_name in ['bull', 'neutral', 'bear']:
            count = regime_dist.get(regime_name, {}).get('count', 0)
            if count < min_regime_trades:
                issues.append(f"Insufficient {regime_name} trades: {count} (minimum {min_regime_trades})")
        
        # Check for regime transitions
        if stats.get('total_transitions', 0) < 20:
            issues.append(f"Insufficient transitions: {stats['total_transitions']} (minimum 20)")
        
        # Check for trend pairs presence
        transition_dist = stats.get('transition_distribution', {})
        trend_pairs = ['neutral→bull', 'bull→neutral', 'neutral→bear', 'bear→neutral']
        missing_pairs = []
        for pair in trend_pairs:
            if pair not in transition_dist or transition_dist[pair]['count'] < 3:
                missing_pairs.append(pair)
        
        if missing_pairs:
            issues.append(f"Missing/insufficient trend pairs: {missing_pairs}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Regime classification validation for correlation analysis: PASSED")
        else:
            logger.warning(f"⚠ Regime classification validation issues: {len(issues)} issues")
        
        return is_valid, issues


def classify_trading_regimes(data: pd.DataFrame, price_precision: float = 1e-6) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to classify trading regimes.
    
    Args:
        data (pd.DataFrame): Trading data with 'price' column
        price_precision (float): Minimum price change to consider non-neutral
        
    Returns:
        Tuple[pd.DataFrame, Dict]: (classified_data, statistics)
    """
    classifier = RegimeClassifier(price_precision=price_precision)
    classified_data = classifier.classify_regimes(data)
    statistics = classifier.get_statistics()
    
    # Validate for correlation analysis
    is_valid, issues = classifier.validate_for_correlation_analysis()
    if not is_valid:
        logger.warning("Regime classification validation warnings:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return classified_data, statistics


if __name__ == "__main__":
    # Test the regime classifier with loaded data
    from data_loader import load_primary_data
    
    file_path = "questdb-query-1751544843847.csv"
    
    try:
        # Load data
        data, metadata = load_primary_data(file_path)
        
        # Classify regimes
        classifier = RegimeClassifier()
        classified_data = classifier.classify_regimes(data)
        
        # Display summary
        print(classifier.get_regime_summary())
        
        # Validate for correlation analysis
        is_valid, issues = classifier.validate_for_correlation_analysis()
        print(f"\nCorrelation Analysis Ready: {'✓ YES' if is_valid else '✗ NO'}")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Show first few classified trades
        print(f"\nFirst 10 Classified Trades:")
        display_cols = ['trade_id', 'price', 'price_change', 'price_return', 'regime_name', 'transition']
        print(classified_data[display_cols].head(10).to_string(index=False))
                
    except Exception as e:
        print(f"Error: {e}")