"""FinBERT-based sentiment analysis for financial news."""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger("CANDLE")


class FinBERTSentimentAnalyzer:
    """Sentiment analysis using FinBERT model."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading FinBERT model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Will use dummy sentiment scores")
            self.model = None
            self.tokenizer = None
        
        # FinBERT labels: 0=positive, 1=negative, 2=neutral
        self.labels = ['positive', 'negative', 'neutral']
    
    def analyze_texts(
        self, 
        texts: List[str], 
        batch_size: int = 16,
        max_length: int = 512
    ) -> List[Dict]:
        """Analyze sentiment of a list of texts."""
        if self.model is None:
            return self._dummy_sentiment(len(texts))
        
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Convert to results
            for j, probs in enumerate(probabilities):
                prob_array = probs.cpu().numpy()
                sentiment_score = prob_array[0] - prob_array[1]  # positive - negative
                
                results.append({
                    'text': batch[j][:100] + "..." if len(batch[j]) > 100 else batch[j],
                    'sentiment_score': float(sentiment_score),  # -1 to 1
                    'positive_prob': float(prob_array[0]),
                    'negative_prob': float(prob_array[1]),
                    'neutral_prob': float(prob_array[2]),
                    'label': self.labels[np.argmax(prob_array)]
                })
        
        return results
    
    def analyze_news_df(self, news_df: pd.DataFrame, text_column: str = 'title') -> pd.DataFrame:
        """Analyze sentiment for news DataFrame and aggregate by date."""
        logger.info(f"Analyzing sentiment for {len(news_df)} news items...")
        
        # Get texts
        texts = news_df[text_column].fillna('').tolist()
        
        # Analyze
        sentiments = self.analyze_texts(texts)
        
        # Add to dataframe
        news_df = news_df.copy()
        news_df['sentiment_score'] = [s['sentiment_score'] for s in sentiments]
        news_df['positive_prob'] = [s['positive_prob'] for s in sentiments]
        news_df['negative_prob'] = [s['negative_prob'] for s in sentiments]
        news_df['neutral_prob'] = [s['neutral_prob'] for s in sentiments]
        news_df['sentiment_label'] = [s['label'] for s in sentiments]
        
        # Aggregate by date
        news_df['date'] = pd.to_datetime(news_df['date'])
        daily_sentiment = news_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'positive_prob': 'mean',
            'negative_prob': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            'date', 'sentiment_mean', 'sentiment_std', 'news_count',
            'positive_prob_mean', 'negative_prob_mean'
        ]
        
        # Create final sentiment features
        daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_mean'].diff(3)
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_mean'].rolling(5).std()
        
        # Set date as index
        daily_sentiment.set_index('date', inplace=True)
        
        # Fill NaN values from rolling operations (don't drop all rows!)
        daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_momentum'].fillna(0)
        daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_volatility'].fillna(
            daily_sentiment['sentiment_std'].mean()
        )
        
        # Only drop rows where we have absolutely no sentiment data
        result = daily_sentiment.dropna(subset=['sentiment_mean'])
        
        logger.info(f"Daily sentiment shape: {result.shape}")
        logger.info(f"Date range: {result.index.min()} to {result.index.max()}")
        return result
    
    def _dummy_sentiment(self, n: int) -> List[Dict]:
        """Generate dummy sentiment for testing without model."""
        logger.warning("Using dummy sentiment scores")
        
        np.random.seed(42)
        return [
            {
                'text': f'dummy_{i}',
                'sentiment_score': np.random.randn() * 0.3,
                'positive_prob': max(0, np.random.randn() * 0.3 + 0.3),
                'negative_prob': max(0, np.random.randn() * 0.3 + 0.2),
                'neutral_prob': 0.4,
                'label': np.random.choice(self.labels)
            }
            for i in range(n)
        ]


def create_sentiment_features(daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Create additional sentiment-based features."""
    features = daily_sentiment.copy()
    
    # Lagged sentiment (effect of yesterday's news on today)
    for lag in [1, 2, 3]:
        features[f'sentiment_lag{lag}'] = features['sentiment_mean'].shift(lag)
        features[f'sentiment_lag{lag}'] = features[f'sentiment_lag{lag}'].fillna(0)
    
    # Moving averages
    features['sentiment_ma5'] = features['sentiment_mean'].rolling(5).mean()
    features['sentiment_ma10'] = features['sentiment_mean'].rolling(10).mean()
    
    # Fill NaN in moving averages with raw sentiment mean
    mean_sentiment = features['sentiment_mean'].mean()
    features['sentiment_ma5'] = features['sentiment_ma5'].fillna(mean_sentiment)
    features['sentiment_ma10'] = features['sentiment_ma10'].fillna(mean_sentiment)
    
    # Extreme sentiment indicators
    features['extreme_positive'] = (features['sentiment_mean'] > 0.5).astype(int)
    features['extreme_negative'] = (features['sentiment_mean'] < -0.5).astype(int)
    
    # Sentiment trend (difference from moving average)
    features['sentiment_trend'] = features['sentiment_mean'] - features['sentiment_ma5']
    
    # Only drop rows with no sentiment_mean (should be none at this point)
    return features.dropna(subset=['sentiment_mean'])
