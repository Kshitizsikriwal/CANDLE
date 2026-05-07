# CANDLE/FCIS - Causal Analysis for Financial Data with Limited Evidence

**Financial Causal Inference System** that moves beyond correlation-based AI to true causal reasoning.

## Overview

Traditional Financial AI answers: *"What will happen?"* (correlation)
**CANDLE** answers: *"Why did it happen?"* and *"What if?"* (causation)

### Example: Ice Cream & Drowning
- **Correlation**: Ice cream sales ↑ → Drowning cases ↑  
- **Causation**: Summer heat causes BOTH (spurious correlation)
- **CANDLE**: Discovers the true causal structure

## Three Pillars

### 1. Causal Discovery (PCMCI+)
Builds causal graphs from time-series data (prices + news sentiment)
- Uses **PCMCI+ algorithm** from tigramite
- Detects lagged causal links: Yesterday's news → Today's returns
- Handles high-dimensional financial data

### 2. Causal Inference (Do-Calculus)
Implements Pearl's **do-calculus** for intervention analysis:
- Estimate `P(Return | do(Fed_Rate = +0.25%))`
- Calculate true causal effects with backdoor adjustment
- Compare: Correlation vs Causation

### 3. Counterfactual Simulation
Answer "What if?" questions:
- *"What if sentiment was very positive on March 15?"*
- *"What if Reliance had crashed -5% instead of +2%?"*
- Backtesting with counterfactual scenarios

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure (edit config.yaml)
cp config.yaml.example config.yaml

# 3. Run full pipeline
python -m src.pipeline
```

## Project Structure

```
causal-3/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── utils.py              # Helper functions
│   ├── data_collection.py    # yfinance + NewsAPI
│   ├── sentiment_analyzer.py # FinBERT sentiment
│   ├── causal_discovery.py   # PCMCI+ algorithm
│   ├── causal_inference.py   # Do-calculus
│   ├── counterfactual.py     # What-if simulation
│   ├── pipeline.py           # Main orchestrator
│   └── visualization.py      # Plotting tools
├── data/
│   ├── raw/                  # Prices, news, sentiment
│   └── processed/            # Merged data, causal graphs
├── results/
│   ├── figures/              # Visualizations
│   └── *.csv                 # Causal effects, counterfactuals
├── notebooks/                # Jupyter tutorials
├── config.yaml               # Configuration
└── requirements.txt          # Dependencies
```

## Configuration

Edit `config.yaml`:

```yaml
data:
  tickers:
    - RELIANCE.NS    # NSE India
    - TCS.NS
    - INFY.NS
  start_date: "2022-01-01"
  end_date: "2024-01-01"
  news_api_key: null  # Get from newsapi.org

causal_discovery:
  pc_alpha: 0.05      # Significance level
  tau_max: 5          # Max time lag

sentiment:
  model_name: "ProsusAI/finbert"
```

## Usage Examples

### Python API

```python
from src.pipeline import CANDLEPipeline

# Initialize and run
pipeline = CANDLEPipeline("config.yaml")
summary = pipeline.run_full_pipeline()

# Access results
print(f"Causal edges: {summary['n_causal_edges']}")
print(f"Data points: {summary['data_points']}")
```

### Individual Components

```python
# Data collection
from src.data_collection import PriceDataCollector

collector = PriceDataCollector(
    tickers=['RELIANCE.NS', 'TCS.NS'],
    start_date='2022-01-01',
    end_date='2024-01-01'
)
prices = collector.fetch_prices()
returns = collector.calculate_returns(prices)

# Sentiment analysis
from src.sentiment_analyzer import FinBERTSentimentAnalyzer

analyzer = FinBERTSentimentAnalyzer()
sentiment = analyzer.analyze_news_df(news_df)

# Causal discovery
from src.causal_discovery import CausalDiscoveryEngine

engine = CausalDiscoveryEngine(pc_alpha=0.05, tau_max=5)
results = engine.discover_causal_graph(data)
graph = results['graph']

# Causal inference
from src.causal_inference import CausalInferenceEngine

inference = CausalInferenceEngine()
inference.fit(data, graph)
effect = inference.estimate_causal_effect(
    treatment='sentiment_mean',
    outcome='RELIANCE',
    intervention_value=1.5
)
print(f"Causal effect: {effect['ate']}")

# Counterfactual simulation
from src.counterfactual import CounterfactualSimulator

sim = CounterfactualSimulator(graph, models, data)
result = sim.simulate_intervention(
    intervention={'sentiment_mean': 1.5},
    target_date='2023-06-15'
)
print(f"Impact: {result['causal_effect']}")
```

## Key Dependencies

- **tigramite**: PCMCI+ causal discovery
- **dowhy**: Do-calculus causal inference
- **transformers**: FinBERT sentiment analysis
- **yfinance**: Stock price data
- **networkx**: Graph operations

## Data Sources

- **Prices**: yfinance (NSE: `.NS` suffix, BSE: `.BO` suffix)
- **News**: NewsAPI (1000 free requests/day) or dummy data
- **Sentiment**: ProsusAI/finbert (HuggingFace)

## Comparison with Other Papers

| Paper | Discovery | Do-Calculus | Time-Series | Counterfactuals |
|-------|-----------|-------------|-------------|-----------------|
| CausalStock | ✓ | ✗ | ✓ | ✗ |
| CSHT | Granger only | ✗ | ✓ | ✗ |
| IRIS | ✓ | ✗ | ✗ | ✗ |
| CGF-LLM | PCMCI | ✗ | ✓ | ✗ |
| **CANDLE** | ✓ **PCMCI+** | ✓ **Backdoor** | ✓ | ✓ |

## Citation

If you use CANDLE in your research, please cite:

```bibtex
@software{candle2024,
  title={CANDLE/FCIS: Causal Analysis for Financial Data},
  author={CANDLE Team},
  year={2024}
}
```

## License

MIT License

## Contributing

Pull requests welcome! Focus areas:
- Better time-series models for counterfactuals
- Additional causal discovery algorithms
- More financial datasets (crypto, forex)

## Contact

For questions or collaborations, open an issue on GitHub.

---

**One Line Summary**: Move Finance AI from "What will happen?" to "Why + What if?" 🎯
