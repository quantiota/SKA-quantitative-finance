"""
SKA Trading Algorithm - Entropy-Based Probability Engine

Structural probability calculation using entropy dynamics for the SKA quantitative finance framework.
Implements the core entropy-based probability formulas that reveal the fundamental laws governing market transitions.

Author: SKA Quantitative Finance Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.stats import entropy as scipy_entropy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntropyProbabilityEngine:
    """
    Entropy-based probability engine for SKA trading algorithm.
    
    Implements the structural probability framework from the mathematical model:
    
    Core Formula: P_t = exp(-|ΔH/H_t|)
    Where:
    - P_t = structural probability of transition at time t
    - ΔH = entropy change between consecutive trades  
    - H_t = current entropy state
    
    Also implements:
    - Quantum state formulation: Ψᵢ = Aᵢ * exp(iHᵢ)
    - Lagrangian dynamics: ΔH = L * Δτ
    - Surface probability aggregation: P_{i→j} = ⟨i|Δτ|j⟩/T
    
    This provides the structural foundation underlying the surface correlation patterns.
    """
    
    def __init__(self, entropy_precision: float = 1e-9):
        """
        Initialize entropy probability engine.
        
        Args:
            entropy_precision (float): Minimum entropy value to avoid division by zero
        """
        self.entropy_precision = entropy_precision
        self.data = None
        self.structural_probabilities = None
        self.quantum_states = None
        self.lagrangian_dynamics = None
        self.statistics = {}
        
    def compute_structural_probabilities(self, classified_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute structural probabilities for all transitions using entropy dynamics.
        
        Args:
            classified_data (pd.DataFrame): Data with entropy and regime information
            
        Returns:
            pd.DataFrame: Data enhanced with structural probability calculations
        """
        logger.info(f"Computing structural probabilities for {len(classified_data)} trades")
        
        # Store data
        self.data = classified_data.copy()
        
        # Calculate entropy changes
        self._calculate_entropy_changes()
        
        # Compute structural probabilities
        self._compute_transition_probabilities()
        
        # Calculate quantum state representations
        self._compute_quantum_states()
        
        # Compute Lagrangian dynamics
        self._compute_lagrangian_dynamics()
        
        # Calculate comprehensive statistics
        self._calculate_statistics()
        
        logger.info("Structural probability computation completed")
        return self.data
    
    def _calculate_entropy_changes(self) -> None:
        """Calculate entropy changes between consecutive trades."""
        # Calculate raw entropy differences
        self.data['entropy_change'] = self.data['entropy'].diff()
        
        # Fill NaN for first trade
        self.data.loc[0, 'entropy_change'] = 0.0
        
        # Calculate absolute entropy changes for probability computation
        self.data['abs_entropy_change'] = self.data['entropy_change'].abs()
        
        # Handle zero/near-zero entropy values for division safety
        self.data['entropy_safe'] = self.data['entropy'].apply(
            lambda x: max(abs(x), self.entropy_precision)
        )
        
        logger.info(f"Entropy changes calculated - Range: {self.data['entropy_change'].min():.6f} to {self.data['entropy_change'].max():.6f}")
    
    def _compute_transition_probabilities(self) -> None:
        """Compute structural probabilities using the core SKA formula."""
        # Core SKA formula: P_t = exp(-|ΔH/H_t|)
        entropy_ratio = self.data['abs_entropy_change'] / self.data['entropy_safe']
        self.data['structural_probability'] = np.exp(-entropy_ratio)
        
        # Alternative formulation for comparison: P_t = exp(-|ΔH|)
        self.data['structural_probability_alt'] = np.exp(-self.data['abs_entropy_change'])
        
        # Normalized probability (0 to 1 range enforcement)
        self.data['structural_probability_norm'] = np.clip(
            self.data['structural_probability'], 0.0, 1.0
        )
        
        # Store main probability series
        self.structural_probabilities = self.data['structural_probability'].values
        
        logger.info(f"Structural probabilities computed - Range: {self.data['structural_probability'].min():.6f} to {self.data['structural_probability'].max():.6f}")
    
    def _compute_quantum_states(self) -> None:
        """Compute quantum state representations for market regimes."""
        # Amplitude calculation: A_i based on entropy magnitude
        self.data['amplitude'] = np.sqrt(self.data['entropy_safe'])
        
        # Phase calculation: exp(iH_i) represented as cos(H) + i*sin(H)
        self.data['phase_real'] = np.cos(self.data['entropy'])
        self.data['phase_imag'] = np.sin(self.data['entropy'])
        
        # Complex quantum state: Ψᵢ = Aᵢ * exp(iHᵢ)
        self.data['quantum_state_real'] = self.data['amplitude'] * self.data['phase_real']
        self.data['quantum_state_imag'] = self.data['amplitude'] * self.data['phase_imag']
        
        # Quantum state magnitude (probability density)
        self.data['quantum_probability_density'] = (
            self.data['quantum_state_real']**2 + self.data['quantum_state_imag']**2
        )
        
        # Store quantum states
        self.quantum_states = {
            'amplitudes': self.data['amplitude'].values,
            'phases_real': self.data['phase_real'].values,
            'phases_imag': self.data['phase_imag'].values,
            'states_real': self.data['quantum_state_real'].values,
            'states_imag': self.data['quantum_state_imag'].values,
            'probability_densities': self.data['quantum_probability_density'].values
        }
        
        logger.info("Quantum state representations computed")
    
    def _compute_lagrangian_dynamics(self) -> None:
        """Compute Lagrangian formulation: ΔH = L * Δτ."""
        # Calculate entropy flow rate L = ΔH / Δτ (avoid division by zero)
        if 'delta_t' in self.data.columns:
            # Use existing delta_t from data
            delta_tau = self.data['delta_t'].replace(0, self.entropy_precision)
        else:
            # Calculate from timestamps
            time_diff = self.data['timestamp'].diff().dt.total_seconds()
            delta_tau = time_diff.fillna(self.entropy_precision).replace(0, self.entropy_precision)
        
        # Lagrangian L = ΔH / Δτ
        self.data['lagrangian'] = self.data['entropy_change'] / delta_tau
        
        # Alternative structural probability using Lagrangian
        # P_t = exp(-|L * Δτ / H_t|) = exp(-|ΔH / H_t|) (same as core formula)
        self.data['structural_probability_lagrangian'] = np.exp(
            -np.abs(self.data['lagrangian'] * delta_tau) / self.data['entropy_safe']
        )
        
        # Store Lagrangian dynamics
        self.lagrangian_dynamics = {
            'lagrangian': self.data['lagrangian'].values,
            'delta_tau': delta_tau.values,
            'entropy_flow_rate': self.data['lagrangian'].values
        }
        
        logger.info(f"Lagrangian dynamics computed - Flow rate range: {self.data['lagrangian'].min():.6f} to {self.data['lagrangian'].max():.6f}")
    
    def _calculate_statistics(self) -> None:
        """Calculate comprehensive entropy probability statistics."""
        # Basic structural probability statistics
        struct_prob = self.data['structural_probability']
        
        # Regime-specific probability analysis
        regime_prob_stats = {}
        for regime in [0, 1, 2]:  # neutral, bull, bear
            regime_mask = self.data['regime'] == regime
            if regime_mask.any():
                regime_probs = struct_prob[regime_mask]
                regime_prob_stats[regime] = {
                    'count': int(regime_mask.sum()),
                    'mean_probability': float(regime_probs.mean()),
                    'std_probability': float(regime_probs.std()),
                    'min_probability': float(regime_probs.min()),
                    'max_probability': float(regime_probs.max())
                }
        
        # Transition-specific probability analysis (9-transition matrix validation)
        transition_prob_stats = {}
        if 'transition' in self.data.columns:
            for transition in self.data['transition'].unique():
                if pd.notna(transition) and transition != 'start→neutral':
                    trans_mask = self.data['transition'] == transition
                    if trans_mask.any():
                        trans_probs = struct_prob[trans_mask]
                        transition_prob_stats[transition] = {
                            'count': int(trans_mask.sum()),
                            'mean_probability': float(trans_probs.mean()),
                            'empirical_frequency': float(trans_mask.sum() / len(self.data))
                        }
        
        # Entropy dynamics analysis
        entropy_stats = {
            'entropy_range': {
                'min': float(self.data['entropy'].min()),
                'max': float(self.data['entropy'].max()),
                'mean': float(self.data['entropy'].mean()),
                'std': float(self.data['entropy'].std())
            },
            'entropy_change_range': {
                'min': float(self.data['entropy_change'].min()),
                'max': float(self.data['entropy_change'].max()),
                'mean': float(self.data['entropy_change'].mean()),
                'std': float(self.data['entropy_change'].std())
            }
        }
        
        # Quantum state analysis
        quantum_stats = {
            'amplitude_range': {
                'min': float(self.data['amplitude'].min()),
                'max': float(self.data['amplitude'].max()),
                'mean': float(self.data['amplitude'].mean())
            },
            'probability_density_range': {
                'min': float(self.data['quantum_probability_density'].min()),
                'max': float(self.data['quantum_probability_density'].max()),
                'mean': float(self.data['quantum_probability_density'].mean())
            }
        }
        
        # Lagrangian analysis
        lagrangian_stats = {
            'flow_rate_range': {
                'min': float(self.data['lagrangian'].min()),
                'max': float(self.data['lagrangian'].max()),
                'mean': float(self.data['lagrangian'].mean()),
                'std': float(self.data['lagrangian'].std())
            }
        }
        
        # Consolidate all statistics
        self.statistics = {
            'structural_probabilities': {
                'overall_mean': float(struct_prob.mean()),
                'overall_std': float(struct_prob.std()),
                'regime_breakdown': regime_prob_stats,
                'transition_breakdown': transition_prob_stats
            },
            'entropy_dynamics': entropy_stats,
            'quantum_states': quantum_stats,
            'lagrangian_dynamics': lagrangian_stats,
            'summary': {
                'total_trades': len(self.data),
                'mean_structural_probability': float(struct_prob.mean()),
                'information_gain_trades': int((self.data['entropy_change'] < 0).sum()),
                'information_loss_trades': int((self.data['entropy_change'] > 0).sum()),
                'neutral_entropy_trades': int((self.data['entropy_change'] == 0).sum())
            }
        }
        
        logger.info(f"Statistics calculated - Mean structural probability: {self.statistics['structural_probabilities']['overall_mean']:.6f}")
    
    def get_structural_probabilities(self) -> np.ndarray:
        """
        Get structural probability series.
        
        Returns:
            np.ndarray: Structural probabilities for all trades
        """
        if self.structural_probabilities is None:
            raise ValueError("Must compute structural probabilities first")
        return self.structural_probabilities.copy()
    
    def get_quantum_states(self) -> Dict:
        """
        Get quantum state representations.
        
        Returns:
            Dict: Complete quantum state data
        """
        if self.quantum_states is None:
            raise ValueError("Must compute quantum states first")
        return self.quantum_states.copy()
    
    def get_lagrangian_dynamics(self) -> Dict:
        """
        Get Lagrangian dynamics data.
        
        Returns:
            Dict: Lagrangian formulation results
        """
        if self.lagrangian_dynamics is None:
            raise ValueError("Must compute Lagrangian dynamics first")
        return self.lagrangian_dynamics.copy()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive entropy probability statistics."""
        return self.statistics.copy()
    
    def validate_ska_framework(self) -> Tuple[bool, List[str]]:
        """
        Validate implementation against SKA mathematical framework.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        if self.data is None:
            return False, ["No entropy probability computation performed"]
        
        # Check for negative entropy values (expected in SKA)
        neg_entropy_count = (self.data['entropy'] < 0).sum()
        if neg_entropy_count == 0:
            issues.append("No negative entropy values found (unusual for SKA framework)")
        
        # Check structural probability range
        prob_min = self.data['structural_probability'].min()
        prob_max = self.data['structural_probability'].max()
        if prob_min < 0 or prob_max > 1:
            issues.append(f"Structural probabilities outside [0,1] range: [{prob_min:.6f}, {prob_max:.6f}]")
        
        # Check for entropy variation
        entropy_std = self.data['entropy'].std()
        if entropy_std < 0.001:
            issues.append(f"Low entropy variation: {entropy_std:.6f} (may affect probability calculation)")
        
        # Check quantum state consistency
        if self.quantum_states is not None:
            # Verify probability density is non-negative
            if (self.quantum_states['probability_densities'] < 0).any():
                issues.append("Negative quantum probability densities detected")
        
        # Validate against known SKA results (9-transition matrix)
        if 'transition' in self.data.columns:
            unique_transitions = self.data['transition'].nunique()
            if unique_transitions < 8:  # Expect at least 8 active transitions
                issues.append(f"Insufficient transition diversity: {unique_transitions} (expected ≥8)")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ SKA framework validation: PASSED")
        else:
            logger.warning(f"⚠ SKA framework validation issues: {len(issues)} issues")
        
        return is_valid, issues
    
    def get_entropy_summary(self) -> str:
        """
        Get formatted summary of entropy probability analysis.
        
        Returns:
            str: Human-readable entropy analysis summary
        """
        if not self.statistics:
            return "No entropy probability analysis performed"
        
        stats = self.statistics
        struct_stats = stats['structural_probabilities']
        entropy_stats = stats['entropy_dynamics']
        summary_stats = stats['summary']
        
        summary = f"""
SKA Entropy Probability Analysis
================================
Total Trades: {summary_stats['total_trades']:,}
Mean Structural Probability: {struct_stats['overall_mean']:.6f}

Entropy Dynamics:
  Range: {entropy_stats['entropy_range']['min']:.6f} to {entropy_stats['entropy_range']['max']:.6f}
  Mean: {entropy_stats['entropy_range']['mean']:.6f}
  Std: {entropy_stats['entropy_range']['std']:.6f}

Information Flow:
  Information Gain Trades: {summary_stats['information_gain_trades']:,} ({100*summary_stats['information_gain_trades']/summary_stats['total_trades']:.1f}%)
  Information Loss Trades: {summary_stats['information_loss_trades']:,} ({100*summary_stats['information_loss_trades']/summary_stats['total_trades']:.1f}%)
  Neutral Entropy Trades: {summary_stats['neutral_entropy_trades']:,} ({100*summary_stats['neutral_entropy_trades']/summary_stats['total_trades']:.1f}%)

Structural Probabilities by Regime:"""
        
        regime_names = {0: 'Neutral', 1: 'Bull', 2: 'Bear'}
        for regime, name in regime_names.items():
            if regime in struct_stats['regime_breakdown']:
                regime_stats = struct_stats['regime_breakdown'][regime]
                summary += f"\n  {name}: {regime_stats['mean_probability']:.6f} ± {regime_stats['std_probability']:.6f} ({regime_stats['count']} trades)"
        
        # Add quantum state information if available
        if 'quantum_states' in stats:
            quantum_stats = stats['quantum_states']
            summary += f"\n\nQuantum State Analysis:"
            summary += f"\n  Amplitude Range: {quantum_stats['amplitude_range']['min']:.6f} to {quantum_stats['amplitude_range']['max']:.6f}"
            summary += f"\n  Mean Probability Density: {quantum_stats['probability_density_range']['mean']:.6f}"
        
        return summary.strip()


def compute_entropy_probabilities(classified_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to compute entropy-based structural probabilities.
    
    Args:
        classified_data (pd.DataFrame): Data with entropy and regime classification
        
    Returns:
        Tuple[pd.DataFrame, Dict]: (enhanced_data, statistics)
    """
    engine = EntropyProbabilityEngine()
    enhanced_data = engine.compute_structural_probabilities(classified_data)
    statistics = engine.get_statistics()
    
    # Validate SKA framework implementation
    is_valid, issues = engine.validate_ska_framework()
    if not is_valid:
        logger.warning("SKA framework validation warnings:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return enhanced_data, statistics


if __name__ == "__main__":
    # Test the entropy probability engine with full pipeline
    from data_loader import load_primary_data
    from regime_classifier import RegimeClassifier
    
    file_path = "questdb-query-1751544843847.csv"
    
    try:
        # Load and process data
        data, metadata = load_primary_data(file_path)
        
        # Classify regimes
        classifier = RegimeClassifier()
        classified_data = classifier.classify_regimes(data)
        
        # Compute entropy probabilities
        engine = EntropyProbabilityEngine()
        enhanced_data = engine.compute_structural_probabilities(classified_data)
        
        # Display results
        print(engine.get_entropy_summary())
        
        # Validate SKA framework
        is_valid, issues = engine.validate_ska_framework()
        print(f"\nSKA Framework Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Show sample entropy probabilities
        print(f"\nSample Entropy Probabilities:")
        display_cols = ['trade_id', 'entropy', 'entropy_change', 'structural_probability', 'regime_name']
        print(enhanced_data[display_cols].head(10).to_string(index=False))
        
        # Show transition probability validation (compare with mathematical model expectations)
        if 'transition' in enhanced_data.columns:
            print(f"\nTransition Structural Probabilities (vs Mathematical Model):")
            transition_stats = engine.get_statistics()['structural_probabilities']['transition_breakdown']
            for transition, stats in list(transition_stats.items())[:5]:
                print(f"  {transition}: {stats['mean_probability']:.6f} (n={stats['count']})")
                
    except Exception as e:
        print(f"Error: {e}")