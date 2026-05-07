#!/usr/bin/env python3
"""
Quick demo of CANDLE/FCIS with minimal data.
This runs a simplified version for testing.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70)
print("CANDLE/FCIS - Quick Demo")
print("=" * 70)

# Step 1: Generate synthetic data
print("\n[1/4] Generating synthetic financial data...")

np.random.seed(42)
n_days = 252  # ~1 year of trading days
dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

# Generate correlated returns
stocks = ['RELIANCE', 'TCS', 'INFY']
returns = pd.DataFrame(index=dates)

# Base market factor
market = np.random.randn(n_days) * 0.01

for stock in stocks:
    # Stock returns = market factor + idiosyncratic noise
    noise = np.random.randn(n_days) * 0.015
    returns[stock] = market * 0.7 + noise

# Generate sentiment based on market (with lag)
sentiment = pd.DataFrame(index=dates)
sentiment['sentiment_mean'] = np.convolve(
    market, [0.3, 0.4, 0.3], mode='same'
) * 10 + np.random.randn(n_days) * 0.5
sentiment['sentiment_std'] = np.abs(np.random.randn(n_days) * 0.3 + 0.5)
sentiment['news_count'] = np.random.poisson(5, n_days)

print(f"  Generated {len(returns)} days of data")
print(f"  Stocks: {stocks}")
print(f"  Sentiment features: {list(sentiment.columns)}")

# Step 2: Merge data
print("\n[2/4] Merging data...")
merged = returns.join(sentiment, how='left').ffill().dropna()
print(f"  Merged data shape: {merged.shape}")

# Step 3: Causal Discovery
print("\n[3/4] Running causal discovery...")

try:
    from causal_discovery import CausalDiscoveryEngine
    
    engine = CausalDiscoveryEngine(
        pc_alpha=0.1,
        tau_max=3,
        tau_min=1
    )
    
    # Normalize for discovery
    normalized = (merged - merged.mean()) / merged.std()
    results = engine.discover_causal_graph(normalized)
    
    graph = results['graph']
    parents = results['parents']
    
    print(f"  Discovered graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print("\n  Causal relationships found:")
    for var, var_parents in parents.items():
        if var_parents:
            for p, lag in var_parents:
                print(f"    {p} (lag {lag}) → {var}")
    
except Exception as e:
    print(f"  Causal discovery skipped: {e}")
    print("  (Install tigramite for full causal discovery)")
    graph = None

# Step 4: Simple Analysis
print("\n[4/4] Analysis Summary")
print("-" * 70)

# Calculate correlations
corr = merged.corr()
print("\nCorrelations with sentiment:")
for stock in stocks:
    print(f"  {stock} ↔ sentiment_mean: {corr.loc[stock, 'sentiment_mean']:.3f}")

# Show some basic causal interpretation
print("\nKey Insight:")
print("  If sentiment_mean ↑ by 1 std:")
for stock in stocks:
    beta = np.polyfit(merged['sentiment_mean'], merged[stock], 1)[0]
    print(f"    {stock} return change: ~{beta:.4f}")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
print("\nTo run full pipeline:")
print("  python run.py")
print("\nTo explore in notebook:")
print("  jupyter notebook notebooks/01_data_collection.ipynb")
