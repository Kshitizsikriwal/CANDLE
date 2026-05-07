"""Tests for data collection module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection import PriceDataCollector, NewsDataCollector


class TestPriceDataCollector:
    """Test price data collection."""
    
    def test_initialization(self):
        collector = PriceDataCollector(
            tickers=['RELIANCE.NS'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        assert collector.tickers == ['RELIANCE.NS']
        assert collector.start_date == '2023-01-01'
    
    def test_calculate_returns(self):
        # Create dummy prices
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'RELIANCE': [100, 101, 102, 101, 103, 104, 105, 104, 106, 107]
        }, index=dates)
        
        collector = PriceDataCollector([], '', '')
        returns = collector.calculate_returns(prices, log_returns=True)
        
        assert len(returns) == 9  # One less due to diff
        assert not returns.isna().any().any()
    
    def test_add_technical_indicators(self):
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = pd.DataFrame({
            'RELIANCE': np.random.randn(50).cumsum() + 100
        }, index=dates)
        
        collector = PriceDataCollector([], '', '')
        features = collector.add_technical_indicators(prices)
        
        assert 'RELIANCE_MA5' in features.columns
        assert 'RELIANCE_MA20' in features.columns
        assert 'RELIANCE_RSI' in features.columns


class TestNewsDataCollector:
    """Test news data collection."""
    
    def test_initialization(self):
        collector = NewsDataCollector(api_key=None)
        assert collector.api_key is None
    
    def test_dummy_news_generation(self):
        collector = NewsDataCollector(api_key=None)
        news = collector._generate_dummy_news()
        
        assert isinstance(news, pd.DataFrame)
        assert len(news) == 500
        assert 'title' in news.columns
        assert 'date' in news.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
