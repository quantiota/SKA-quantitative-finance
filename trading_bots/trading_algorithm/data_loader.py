"""
SKA Trading Algorithm - Data Loader Module

Primary CSV data processing and validation for the SKA quantitative finance framework.
Handles loading, cleaning, and validation of the primary trading data file.

Author: SKA Quantitative Finance Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Primary CSV data loader and validator for SKA trading algorithm.
    
    Based on actual CSV structure analysis:
    - 16 columns total
    - Empty values in: buyer_order_id, seller_order_id, entropy (first row), delta_h_over_delta_d
    - 3,584 data rows + 1 header row
    """
    
    # All columns in the CSV
    ALL_COLUMNS = [
        'symbol', 'trade_id', 'price', 'quantity', 'buyer_order_id', 'seller_order_id',
        'is_buyer_market_maker', 'knowledge', 'decision', 'decision_norm', 'entropy',
        'delta_t', 'cosine_similarity', 'delta_h_over_delta_d', 'frobenius_norm', 'timestamp'
    ]
    
    # Essential columns for SKA analysis
    ESSENTIAL_COLUMNS = [
        'symbol', 'trade_id', 'price', 'quantity', 'knowledge', 'decision', 
        'decision_norm', 'entropy', 'delta_t', 'timestamp'
    ]
    
    # Columns that can have empty/NaN values
    OPTIONAL_COLUMNS = [
        'buyer_order_id', 'seller_order_id', 'delta_h_over_delta_d'
    ]
    
    def __init__(self, file_path: str):
        """
        Initialize data loader with file path.
        
        Args:
            file_path (str): Path to primary CSV file
        """
        self.file_path = file_path
        self.data = None
        self.metadata = {}
        
    def load_and_validate(self) -> pd.DataFrame:
        """
        Load CSV data and perform validation based on actual file structure.
        
        Returns:
            pd.DataFrame: Validated and cleaned trading data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data structure is invalid
        """
        logger.info(f"Loading data from {self.file_path}")
        
        try:
            # Load CSV data
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(self.data)} records")
            
            # Validate and process
            self._validate_structure()
            self._process_data_types()
            self._handle_missing_values()
            self._sort_and_clean()
            self._extract_metadata()
            
            logger.info("Data validation completed successfully")
            return self.data
            
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_structure(self) -> None:
        """Validate CSV structure matches expected format."""
        # Check all expected columns are present
        missing_columns = [col for col in self.ALL_COLUMNS if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")
        
        extra_columns = [col for col in self.data.columns if col not in self.ALL_COLUMNS]
        if extra_columns:
            logger.warning(f"Extra columns found (will be ignored): {extra_columns}")
        
        logger.info(f"CSV structure validated: {len(self.data.columns)} columns, {len(self.data)} rows")
    
    def _process_data_types(self) -> None:
        """Convert columns to appropriate data types."""
        try:
            # Convert timestamp
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            
            # Convert trade_id to int64 (handle scientific notation)
            self.data['trade_id'] = self.data['trade_id'].astype('int64')
            
            # Convert core numeric columns
            numeric_cols = ['price', 'quantity', 'knowledge', 'decision', 'decision_norm', 'delta_t']
            for col in numeric_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Convert entropy separately (has empty values in first row)
            self.data['entropy'] = pd.to_numeric(self.data['entropy'], errors='coerce')
            
            # Convert optional numeric columns
            optional_numeric = ['cosine_similarity', 'delta_h_over_delta_d', 'frobenius_norm']
            for col in optional_numeric:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Convert boolean column
            self.data['is_buyer_market_maker'] = self.data['is_buyer_market_maker'].astype('bool')
            
            logger.info("Data type conversion completed")
            
        except Exception as e:
            raise ValueError(f"Data type conversion failed: {str(e)}")
    
    def _handle_missing_values(self) -> None:
        """Handle missing values appropriately."""
        initial_count = len(self.data)
        
        # Handle entropy missing in first row (interpolate from second row)
        if pd.isna(self.data.loc[0, 'entropy']) and len(self.data) > 1:
            # Use a small initial entropy value
            self.data.loc[0, 'entropy'] = 0.001
            logger.info("Filled missing entropy in first row with initial value")
        
        # Remove rows where essential numeric columns are NaN
        essential_numeric = ['trade_id', 'price', 'entropy', 'knowledge', 'decision']
        before_filter = len(self.data)
        self.data = self.data.dropna(subset=essential_numeric)
        after_filter = len(self.data)
        
        if before_filter > after_filter:
            logger.warning(f"Removed {before_filter - after_filter} rows with missing essential data")
        
        logger.info(f"Missing value handling completed: {len(self.data)} records retained")
    
    def _sort_and_clean(self) -> None:
        """Sort data and remove duplicates."""
        # Sort by trade_id (chronological order)
        self.data = self.data.sort_values('trade_id').reset_index(drop=True)
        
        # Remove duplicate trade_ids
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=['trade_id'])
        duplicate_count = initial_count - len(self.data)
        
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate trade_ids")
        
        logger.info(f"Data sorting and cleaning completed: {len(self.data)} final records")
    
    def _extract_metadata(self) -> None:
        """Extract comprehensive metadata about the dataset."""
        if self.data is None or len(self.data) == 0:
            self.metadata = {}
            return
        
        self.metadata = {
            'symbol': str(self.data['symbol'].iloc[0]),
            'total_trades': len(self.data),
            'first_trade_id': int(self.data['trade_id'].min()),
            'last_trade_id': int(self.data['trade_id'].max()),
            'first_timestamp': self.data['timestamp'].min(),
            'last_timestamp': self.data['timestamp'].max(),
            'time_span_seconds': (self.data['timestamp'].max() - self.data['timestamp'].min()).total_seconds(),
            'price_stats': {
                'min': float(self.data['price'].min()),
                'max': float(self.data['price'].max()),
                'mean': float(self.data['price'].mean()),
                'std': float(self.data['price'].std())
            },
            'entropy_stats': {
                'min': float(self.data['entropy'].min()),
                'max': float(self.data['entropy'].max()),
                'mean': float(self.data['entropy'].mean()),
                'std': float(self.data['entropy'].std()),
                'nan_count': int(self.data['entropy'].isna().sum())
            },
            'data_quality': {
                'missing_entropy': int(self.data['entropy'].isna().sum()),
                'missing_delta_h': int(self.data['delta_h_over_delta_d'].isna().sum()) if 'delta_h_over_delta_d' in self.data.columns else 0,
                'zero_prices': int((self.data['price'] == 0).sum()),
                'negative_entropy': int((self.data['entropy'] < 0).sum())
            }
        }
        
        logger.info(f"Metadata extracted for {self.metadata['symbol']}: {self.metadata['total_trades']} trades")
        logger.info(f"Price range: ${self.metadata['price_stats']['min']:.4f} - ${self.metadata['price_stats']['max']:.4f}")
        logger.info(f"Entropy range: {self.metadata['entropy_stats']['min']:.6f} - {self.metadata['entropy_stats']['max']:.6f}")
    
    def get_metadata(self) -> Dict:
        """Get dataset metadata."""
        return self.metadata.copy()
    
    def get_data_summary(self) -> str:
        """Get formatted summary of the loaded data."""
        if not self.metadata:
            return "No data loaded"
        
        summary = f"""
SKA Trading Data Summary
========================
Symbol: {self.metadata['symbol']}
Total Trades: {self.metadata['total_trades']:,}
Trade ID Range: {self.metadata['first_trade_id']:,} - {self.metadata['last_trade_id']:,}
Time Span: {self.metadata['time_span_seconds']:.1f} seconds ({self.metadata['time_span_seconds']/3600:.2f} hours)
Date Range: {self.metadata['first_timestamp'].strftime('%Y-%m-%d %H:%M:%S')} to {self.metadata['last_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Price Statistics:
  Min: ${self.metadata['price_stats']['min']:.4f}
  Max: ${self.metadata['price_stats']['max']:.4f}
  Mean: ${self.metadata['price_stats']['mean']:.4f}
  Std: ${self.metadata['price_stats']['std']:.4f}

Entropy Statistics:
  Min: {self.metadata['entropy_stats']['min']:.6f}
  Max: {self.metadata['entropy_stats']['max']:.6f}
  Mean: {self.metadata['entropy_stats']['mean']:.6f}
  Std: {self.metadata['entropy_stats']['std']:.6f}

Data Quality:
  Missing entropy values: {self.metadata['data_quality']['missing_entropy']}
  Missing delta_h values: {self.metadata['data_quality']['missing_delta_h']}
  Zero prices: {self.metadata['data_quality']['zero_prices']}
  Negative entropy: {self.metadata['data_quality']['negative_entropy']}
        """
        return summary.strip()
    
    def validate_for_ska_analysis(self) -> Tuple[bool, List[str]]:
        """Validate data specifically for SKA analysis requirements."""
        issues = []
        
        if self.data is None:
            return False, ["No data loaded"]
        
        # Check minimum data requirements
        if len(self.data) < 100:
            issues.append(f"Insufficient data: {len(self.data)} trades (minimum 100 required)")
        
        # Note: Negative entropy is valid in SKA framework (represents information gain)
        # Only flag if ALL entropy values are the same (no variation)
        if self.data['entropy'].nunique() <= 1:
            issues.append("No entropy variation detected (all values identical)")
        
        # Check for sufficient price variation
        price_std = self.data['price'].std()
        if price_std < 0.0001:
            issues.append(f"Low price variation: {price_std:.6f}")
        
        # Check for sufficient entropy variation
        entropy_std = self.data['entropy'].std()
        if entropy_std < 0.001:
            issues.append(f"Low entropy variation: {entropy_std:.6f}")
        
        # Check timestamp ordering
        if not self.data['trade_id'].is_monotonic_increasing:
            issues.append("Trade IDs not in increasing order")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Data validation for SKA analysis: PASSED")
        else:
            logger.warning(f"⚠ Data validation issues found: {len(issues)} issues")
        
        return is_valid, issues


def load_primary_data(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to load and validate primary CSV data.
    
    Args:
        file_path (str): Path to primary CSV file
        
    Returns:
        Tuple[pd.DataFrame, Dict]: (validated_data, metadata)
    """
    loader = DataLoader(file_path)
    data = loader.load_and_validate()
    metadata = loader.get_metadata()
    
    # Validate for SKA analysis
    is_valid, issues = loader.validate_for_ska_analysis()
    if not is_valid:
        logger.warning("Data validation warnings:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return data, metadata


if __name__ == "__main__":
    # Test the data loader with the primary CSV file
    file_path = "questdb-query-1751544843847.csv"
    
    try:
        data, metadata = load_primary_data(file_path)
        
        loader = DataLoader(file_path)
        loader.data = data
        loader.metadata = metadata
        
        print(loader.get_data_summary())
        
        is_valid, issues = loader.validate_for_ska_analysis()
        print(f"\nSKA Analysis Ready: {'✓ YES' if is_valid else '✗ NO'}")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
                
    except Exception as e:
        print(f"Error: {e}")