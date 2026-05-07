"""Data collection module for CANDLE framework.

Collects stock price data using yfinance and news data via NewsAPI.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger("CANDLE")


class PriceDataCollector:
    """Collects historical stock price data."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_prices(self, interval: str = "1d") -> pd.DataFrame:
        """Fetch closing prices for all tickers."""
        logger.info(f"Fetching price data for {len(self.tickers)} stocks...")
        
        df = yf.download(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date,
            interval=interval,
            progress=True,
            auto_adjust=True
        )
        
        # Extract closing prices
        if len(self.tickers) == 1:
            prices = df['Close'].to_frame(self.tickers[0])
        else:
            prices = df['Close']
        
        # Clean column names (remove .NS, .BO suffixes)
        prices.columns = [c.replace('.NS', '').replace('.BO', '') 
                         for c in prices.columns]
        
        logger.info(f"Price data shape: {prices.shape}")
        logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        
        return prices
    
    def calculate_returns(self, prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
        """Calculate returns from prices."""
        if log_returns:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        return returns.dropna()
    
    def add_technical_indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as additional features."""
        features = prices.copy()
        
        for col in prices.columns:
            # Moving averages
            features[f'{col}_MA5'] = prices[col].rolling(5).mean()
            features[f'{col}_MA20'] = prices[col].rolling(20).mean()
            
            # Volatility (rolling std)
            features[f'{col}_VOL20'] = prices[col].rolling(20).std()
            
            # RSI calculation
            delta = prices[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features[f'{col}_RSI'] = 100 - (100 / (1 + rs))
        
        return features.dropna()


class NewsDataCollector:
    """Collects financial news data."""
    
    def __init__(self, api_key: Optional[str] = None, 
                 start_date: Optional[str] = None, 
                 end_date: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_news(
        self, 
        query: str = "stock market OR finance OR economy",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        language: str = "en",
        max_results: int = 1000
    ) -> pd.DataFrame:
        """Fetch news articles from NewsAPI."""
        if not self.api_key:
            logger.warning("No API key provided. Using dummy news data.")
            return self._generate_dummy_news()
        
        logger.info("Fetching news data...")
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        articles = []
        page = 1
        
        while len(articles) < max_results:
            params = {
                'q': query,
                'from': from_date,
                'to': to_date,
                'language': language,
                'sortBy': 'publishedAt',
                'apiKey': self.api_key,
                'pageSize': 100,
                'page': page
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if data.get('status') != 'ok':
                    break
                
                new_articles = data.get('articles', [])
                if not new_articles:
                    break
                
                articles.extend(new_articles)
                page += 1
                
                if len(articles) >= max_results:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching news: {e}")
                break
        
        # Convert to DataFrame
        news_df = pd.DataFrame([
            {
                'date': datetime.strptime(a['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').date(),
                'title': a['title'],
                'description': a.get('description', ''),
                'content': a.get('content', ''),
                'source': a['source']['name']
            }
            for a in articles[:max_results]
        ])
        
        logger.info(f"Fetched {len(news_df)} news articles")
        return news_df
    
    def _generate_dummy_news(self) -> pd.DataFrame:
        """Generate dummy news data for testing, aligned with price dates."""
        logger.info("Generating dummy news data...")
        
        dummy_headlines = [
            "Market rallies as inflation data shows improvement",
            "Fed signals potential rate cuts in coming months",
            "Tech stocks surge on AI optimism",
            "Banking sector faces regulatory scrutiny",
            "Oil prices fluctuate amid geopolitical tensions",
            "India's GDP growth exceeds expectations",
            "RBI maintains repo rate in monetary policy",
            "IT sector sees strong quarterly earnings",
            "Foreign investment flows into emerging markets",
            "Real estate sector shows recovery signs",
            "Inflation concerns weigh on market sentiment",
            "Tech sector faces headwinds from global slowdown",
            "Energy prices stabilize after volatile week",
            "Consumer spending remains resilient despite inflation",
            "Manufacturing PMI shows expansionary trend"
        ] * 40  # 600 entries
        
        # Use configured date range if available, otherwise use recent dates
        if self.start_date and self.end_date:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            # Generate dates within the range
            n_days = (end - start).days + 1
            dates = pd.date_range(start=start, periods=min(n_days, len(dummy_headlines)), freq='D')
            # Extend with random sampling if needed
            if len(dates) < len(dummy_headlines):
                extra_dates = pd.date_range(start=start, end=end, periods=len(dummy_headlines) - len(dates))
                dates = dates.append(extra_dates)
            else:
                dates = dates[:len(dummy_headlines)]
            logger.info(f"Generating dummy news for date range: {start.date()} to {end.date()}")
        else:
            dates = pd.date_range(end=datetime.now(), periods=len(dummy_headlines), freq='D')
        
        news_df = pd.DataFrame({
            'date': dates.date,
            'title': dummy_headlines[:len(dates)],
            'description': dummy_headlines[:len(dates)],
            'content': dummy_headlines[:len(dates)],
            'source': ['Financial News'] * len(dates)
        })
        
        return news_df


def merge_price_and_sentiment(
    prices: pd.DataFrame, 
    sentiment: pd.DataFrame,
    logger=None
) -> pd.DataFrame:
    """Merge price data with daily sentiment scores."""
    # Ensure both have datetime index (normalize to date only)
    prices.index = pd.to_datetime(prices.index).normalize()
    sentiment.index = pd.to_datetime(sentiment.index).normalize()
    
    if logger:
        logger.info(f"Prices index range: {prices.index.min()} to {prices.index.max()} ({len(prices)} rows)")
        logger.info(f"Sentiment index range: {sentiment.index.min()} to {sentiment.index.max()} ({len(sentiment)} rows)")
        # Check overlap
        price_dates = set(prices.index)
        sentiment_dates = set(sentiment.index)
        overlap = price_dates & sentiment_dates
        logger.info(f"Overlapping dates: {len(overlap)}")
    
    # Merge on date - use outer join to keep all dates, then filter
    merged = prices.join(sentiment, how='outer')
    
    # Sort by date
    merged = merged.sort_index()
    
    # Fill missing sentiment with forward fill, then backward fill
    sentiment_cols = [c for c in sentiment.columns if 'sentiment' in c.lower() or 'news' in c.lower()]
    for col in sentiment_cols:
        merged[col] = merged[col].ffill().bfill()
    
    # Fill missing price data (if any) - only for price columns
    price_cols = [c for c in prices.columns]
    for col in price_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()
    
    # Remove rows where we don't have both price and sentiment
    # Keep rows that have at least some price data (stocks) and some sentiment
    stock_cols = [c for c in prices.columns if c not in sentiment.columns]
    merged = merged.dropna(subset=stock_cols[:1] + sentiment_cols[:1])
    
    return merged
