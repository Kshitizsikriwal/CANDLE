"""Utility functions for CANDLE framework."""

import os
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(log_path: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    Path(log_path).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_path, f"candle_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("CANDLE")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if set
    if os.getenv('NEWS_API_KEY'):
        config['data']['news_api_key'] = os.getenv('NEWS_API_KEY')
    
    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories."""
    paths = [
        config['data']['raw_data_path'],
        config['data']['processed_data_path'],
        config['output']['results_path'],
        config['output']['figures_path'],
        config['output']['logs_path']
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def z_normalize(df):
    """Z-normalize dataframe (subtract mean, divide by std)."""
    return (df - df.mean()) / df.std()


def clean_ticker_names(df):
    """Remove .NS and .BO suffixes from column names."""
    df = df.copy()
    df.columns = [c.replace('.NS', '').replace('.BO', '') for c in df.columns]
    return df


def get_common_nse_tickers():
    """Return list of common NSE stock tickers with .NS suffix."""
    return [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'WIPRO.NS', 'SBIN.NS', 'TATAMOTORS.NS', 'ADANIENT.NS', 'BAJFINANCE.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SUNPHARMA.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'MARUTI.NS', 'AXISBANK.NS', 'LT.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS'
    ]
