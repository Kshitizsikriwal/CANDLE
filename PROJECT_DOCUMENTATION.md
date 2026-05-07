# CANDLE/FCIS - Complete Project Documentation

## Project Overview

**CANDLE/FCIS: Causal Analysis for Financial Data with Limited Evidence / Financial Causal Inference System**

A comprehensive framework for causal inference in financial markets that moves beyond traditional correlation-based AI to true causal reasoning. The system implements Pearl's causal inference framework combined with modern machine learning to answer "Why did it happen?" and "What if?" questions in financial contexts.

### Core Philosophy

Traditional Financial AI answers: *"What will happen?"* (correlation-based prediction)
**CANDLE** answers: *"Why did it happen?"* and *"What if?"* (causal reasoning)

### Motivating Example: Ice Cream & Drowning
- **Correlation**: Ice cream sales ↑ → Drowning cases ↑ (spurious correlation)
- **Causation**: Summer heat causes BOTH (hidden confounder)
- **CANDLE**: Discovers the true causal structure by identifying confounders

---

## Three Pillars of CANDLE

### 1. Causal Discovery (PCMCI+)
**Building causal graphs from time-series data**

- **Algorithm**: PCMCI+ (Peter-Clark Momentary Conditional Independence)
- **Implementation**: Uses tigramite library for time-series causal discovery
- **Purpose**: Detects lagged causal links (e.g., Yesterday's news → Today's returns)
- **Fallback**: Granger causality when tigramite unavailable
- **Key Parameters**:
  - `pc_alpha`: Significance level (default: 0.05)
  - `tau_max`: Maximum time lag (default: 5 days)
  - `tau_min`: Minimum time lag (default: 1 day)

**Output**: Directed acyclic graph (DAG) showing causal relationships between variables with time lags.

### 2. Causal Inference (Do-Calculus)
**Estimating causal effects using Pearl's framework**

- **Method**: Backdoor criterion adjustment
- **Implementation**: Custom implementation (DoWhy disabled due to Python 3.13 compatibility)
- **Purpose**: Estimate P(Return | do(Fed_Rate = +0.25%)) - true causal effect
- **Key Features**:
  - Automatic adjustment set detection
  - Bootstrap confidence intervals
  - Comparison: Correlation vs Causation
- **Model Types**: Ridge regression (default), Linear, Lasso, Random Forest, Gradient Boosting

**Output**: Average Treatment Effect (ATE) with confidence intervals.

### 3. Counterfactual Simulation
**Answering "What if?" questions**

- **Purpose**: Simulate alternative scenarios
- **Use Cases**:
  - Historical event replay
  - Trading decision evaluation
  - Scenario stress testing
- **Features**:
  - Single intervention simulation
  - What-if analysis across value ranges
  - Batch counterfactual scenarios
  - Hindsight analysis for past decisions

**Output**: Counterfactual predictions, causal impact scores, most affected variables.

---

## Data Sources

### 1. Stock Price Data

**Source**: Yahoo Finance via yfinance library

**Data Collected**:
- Daily closing prices
- Adjusted for splits and dividends
- Multiple stock tickers simultaneously

**Tickers Used** (NSE India):
- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- HDFCBANK.NS (HDFC Bank)
- ICICIBANK.NS (ICICI Bank)
- WIPRO.NS (Wipro)
- SBIN.NS (State Bank of India)
- MARUTI.NS (Maruti Suzuki)
- ADANIENT.NS (Adani Enterprises)
- BAJFINANCE.NS (Bajaj Finance)

**Date Range**: January 1, 2016 to January 1, 2026 (includes COVID-19 period 2020-2021)

**Data Processing**:
- Calculate log returns: `log(P_t / P_{t-1})`
- Handle missing data with forward/backward fill
- Z-normalization for causal discovery

### 2. News Data

**Primary Source**: NewsAPI (newsapi.org)
- **API Key**: Free tier (1000 requests/day)
- **Query**: "stock market India OR NSE OR finance OR economy"
- **Language**: English
- **Fields**: Title, description, content, source, publication date

**Fallback Source**: Dummy news data
- Generated when no API key provided
- 15 realistic financial headlines repeated
- Aligned with price date range
- Used for testing and demonstration

**Data Processing**:
- Clean text (remove special characters, lowercase)
- Aggregate by date
- Handle missing dates

### 3. Sentiment Analysis

**Model**: FinBERT (ProsusAI/finbert)
- **Framework**: HuggingFace Transformers
- **Architecture**: BERT fine-tuned on financial text
- **Labels**: Positive (0), Negative (1), Neutral (2)
- **Output**: Sentiment score (-1 to +1)

**Fallback**: Dummy sentiment when model unavailable
- Random sentiment scores with financial context
- Used for testing without GPU

**Features Engineered**:
- `sentiment_mean`: Daily average sentiment
- `sentiment_std`: Daily sentiment volatility
- `news_count`: Number of articles per day
- `positive_prob_mean`: Average positive probability
- `negative_prob_mean`: Average negative probability
- `sentiment_momentum`: 3-day sentiment change
- `sentiment_volatility`: 5-day rolling std
- `sentiment_lag1`, `sentiment_lag2`, `sentiment_lag3`: Lagged sentiment
- `sentiment_ma5`, `sentiment_ma10`: Moving averages
- `extreme_positive`: Binary indicator (sentiment > 0.5)
- `extreme_negative`: Binary indicator (sentiment < -0.5)
- `sentiment_trend`: Difference from moving average

---

## Implementation Architecture

### Project Structure

```
causal-3/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Main entry point
│   ├── utils.py                 # Utility functions
│   ├── data_collection.py       # Price and news collection
│   ├── sentiment_analyzer.py    # FinBERT sentiment analysis
│   ├── causal_discovery.py      # PCMCI+ causal discovery
│   ├── causal_inference.py      # Do-calculus inference
│   ├── counterfactual.py        # Counterfactual simulation
│   ├── pipeline.py              # Main pipeline orchestrator
│   ├── visualization.py         # Plotting and visualization
│   └── architecture_viz.py      # Architecture visualization
├── data/                         # Data directory
│   ├── raw/                     # Raw data
│   │   ├── prices.csv           # Stock prices
│   │   ├── returns.csv          # Log returns
│   │   ├── news.csv             # News articles
│   │   └── sentiment_daily.csv  # Daily sentiment
│   └── processed/               # Processed data
│       ├── merged_data.csv      # Prices + sentiment merged
│       ├── merged_data_normalized.csv  # Z-normalized
│       ├── causal_graph.graphml # Causal graph (GraphML)
│       ├── causal_graph.pkl     # Causal graph (Pickle)
│       └── causal_parents.csv   # Parent relationships
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_collection.ipynb        # Data collection tutorial
│   ├── 02_causal_discovery.ipynb       # Causal discovery tutorial
│   ├── 03_causal_inference.ipynb       # Causal inference tutorial
│   ├── 04_counterfactual.ipynb         # Counterfactual tutorial
│   └── 05_architecture_visualization.ipynb  # Architecture viz
├── results/                      # Results directory
│   ├── figures/                 # Visualizations
│   │   ├── causal_graph.png
│   │   ├── causal_effects_heatmap.png
│   │   ├── counterfactual_comparison.png
│   │   └── what_if_analysis.png
│   ├── causal_effects.csv       # Causal effect estimates
│   ├── causal_effects_all.csv   # Multiple effects
│   └── counterfactual_scenarios.csv  # Counterfactual results
├── logs/                         # Log files
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_data_collection.py
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── Makefile                     # Build automation
├── README.md                    # Project README
├── run.py                       # Quick run script
└── demo.py                      # Demo script
```

### Module Details

#### 1. data_collection.py

**Classes**:
- `PriceDataCollector`: Fetches stock prices from Yahoo Finance
  - `fetch_prices()`: Download historical prices
  - `calculate_returns()`: Compute log returns
  - `add_technical_indicators()`: Add MA, RSI, volatility

- `NewsDataCollector`: Fetches news from NewsAPI
  - `fetch_news()`: Download news articles
  - `_generate_dummy_news()`: Fallback dummy data

**Functions**:
- `merge_price_and_sentiment()`: Merge price and sentiment data by date

**Key Features**:
- Handles missing data with ffill/bfill
- Normalizes ticker names (removes .NS, .BO)
- Supports multiple tickers simultaneously
- Progress bars for large downloads

#### 2. sentiment_analyzer.py

**Classes**:
- `FinBERTSentimentAnalyzer`: Sentiment analysis using FinBERT
  - `analyze_texts()`: Analyze sentiment of text list
  - `analyze_news_df()`: Analyze and aggregate by date
  - `_dummy_sentiment()`: Fallback dummy sentiment

**Functions**:
- `create_sentiment_features()`: Engineer additional sentiment features

**Key Features**:
- Batch processing for efficiency
- GPU acceleration when available
- Sentiment score: positive_prob - negative_prob
- Daily aggregation with statistics
- Lag features for causal discovery
- Extreme sentiment indicators

#### 3. causal_discovery.py

**Classes**:
- `CausalDiscoveryEngine`: Causal discovery using PCMCI+
  - `discover_causal_graph()`: Main discovery method
  - `_discover_with_pcmci()`: Using tigramite
  - `_discover_fallback()`: Using Granger causality
  - `get_causal_order()`: Topological ordering
  - `get_markov_blanket()`: Markov blanket of variable

**Key Features**:
- Supports multiple independence tests (ParCorr, GPDC, CMIknn)
- Handles cycles with degree-based ordering
- Converts to NetworkX for easy manipulation
- Extracts parent relationships
- Saves in multiple formats (GraphML, Pickle, CSV)

**Algorithm**:
1. **PCMCI+** (preferred): Condition selection + Momentary Conditional Independence
   - More powerful than Granger causality
   - Handles contemporaneous links
   - Robust to high-dimensional data

2. **Granger Causality** (fallback): Time-lagged correlation
   - Simpler, faster
   - Only lagged links
   - F-statistic significance test

#### 4. causal_inference.py

**Classes**:
- `CausalInferenceEngine`: Do-calculus implementation
  - `fit()`: Fit outcome models for all variables
  - `estimate_causal_effect()`: Estimate ATE for single treatment
  - `estimate_all_effects()`: Batch effect estimation
  - `_find_adjustment_set()`: Backdoor criterion
  - `_fit_outcome_model()`: Train regression models

**Key Features**:
- Custom backdoor adjustment implementation
- Bootstrap confidence intervals (100 samples)
- Automatic adjustment set detection
- Multiple model types (Ridge, Linear, Lasso, RF, GB)
- Individual Treatment Effects (ITE) estimation
- DoWhy disabled (Python 3.13 compatibility issues)

**Method**: Backdoor Adjustment
1. Identify adjustment set (confounders)
2. Fit outcome model: E[Y | X, Z]
3. Estimate E[Y | do(X=x)] = Σ_z E[Y | X=x, Z=z] P(Z=z)
4. Compute ATE = E[Y | do(X=1)] - E[Y | do(X=0)]

#### 5. counterfactual.py

**Classes**:
- `CounterfactualSimulator`: Counterfactual simulation
  - `simulate_intervention()`: Single intervention
  - `what_if_analysis()`: Range of values
  - `hindsight_analysis()`: Historical decision evaluation
  - `batch_counterfactuals()`: Multiple scenarios

**Key Features**:
- Topological propagation through causal graph
- Handles multiple simultaneous interventions
- Causal impact scoring (L2 norm)
- Identifies most affected variables
- Time horizon propagation (multi-step)

**Algorithm**:
1. Apply intervention (break incoming links)
2. Propagate effects in topological order
3. For each variable: predict using parent values
4. Compare with factual values
5. Compute impact scores

#### 6. pipeline.py

**Classes**:
- `CANDLEPipeline`: Main pipeline orchestrator
  - `run_data_collection()`: Step 1
  - `run_sentiment_analysis()`: Step 2
  - `merge_and_preprocess()`: Step 3
  - `run_causal_discovery()`: Step 4
  - `run_causal_inference()`: Step 5
  - `run_counterfactual_simulation()`: Step 6
  - `run_full_pipeline()`: Execute all steps

**Key Features**:
- Automated end-to-end execution
- Progress logging at each step
- Automatic directory creation
- Error handling and fallbacks
- Summary statistics generation

#### 7. visualization.py

**Classes**:
- `CANDLEVisualizer`: Visualization utilities
  - `plot_causal_graph()`: Network graph visualization
  - `plot_causal_effects()`: Heatmap of effects
  - `plot_counterfactual_comparison()`: Factual vs counterfactual
  - `plot_what_if_analysis()`: What-if curve
  - `plot_sentiment_time_series()`: Sentiment over time

**Key Features**:
- NetworkX graph visualization
- Matplotlib/Seaborn plots
- High-DPI output (300 dpi)
- Automatic figure saving
- Customizable styling

---

## Configuration

### config.yaml

```yaml
# Data Settings
data:
  tickers:
    - RELIANCE.NS
    - TCS.NS
    - INFY.NS
    - HDFCBANK.NS
    - ICICIBANK.NS
    - WIPRO.NS
    - SBIN.NS
    - MARUTI.NS
    - ADANIENT.NS
    - BAJFINANCE.NS
  start_date: "2016-01-01"
  end_date: "2026-01-01"
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  news_api_key: null  # Set via environment variable NEWS_API_KEY

# Causal Discovery Settings
causal_discovery:
  algorithm: "pcmci_plus"
  pc_alpha: 0.05      # Significance level
  tau_max: 5          # Max time lag
  tau_min: 1          # Min time lag

# FinBERT Sentiment Analysis
sentiment:
  model_name: "ProsusAI/finbert"
  batch_size: 32
  max_length: 512

# Inference Settings
inference:
  method: "backdoor"  # backdoor, frontdoor, iv
  confidence_level: 0.95

# Counterfactual Settings
counterfactual:
  num_samples: 1000
  method: "linear"

# Output
output:
  results_path: "results"
  figures_path: "results/figures"
  logs_path: "logs"
```

### Environment Variables

Create `.env` file (copy from `.env.example`):
```
NEWS_API_KEY=your_api_key_here
```

Get free API key from: https://newsapi.org/

---

## Dependencies

### requirements.txt

```
yfinance>=0.2.28              # Stock price data
pandas>=2.0.0                # Data manipulation
numpy>=1.24.0                 # Numerical computing
scikit-learn>=1.3.0           # Machine learning
tigramite>=5.2                # Causal discovery
networkx>=3.1                 # Graph operations
matplotlib>=3.7.0             # Plotting
seaborn>=0.12.0              # Statistical visualization
transformers>=4.30.0          # FinBERT model
torch>=2.0.0                  # PyTorch (for transformers)
requests>=2.31.0              # HTTP requests
python-dotenv>=1.0.0          # Environment variables
dowhy>=0.9.1                  # Causal inference (disabled)
statsmodels>=0.14.0           # Statistical tests
plotly>=5.15.0                # Interactive plots
tqdm>=4.65.0                  # Progress bars
```

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp config.yaml.example config.yaml
cp .env.example .env
# Edit .env to add NEWS_API_KEY
```

---

## Usage

### 1. Run Full Pipeline

```bash
# Using Python module
python -m src.pipeline

# Or using run.py
python run.py
```

This executes all 6 steps:
1. Data collection (prices + news)
2. Sentiment analysis (FinBERT)
3. Data merging & preprocessing
4. Causal discovery (PCMCI+)
5. Causal inference (do-calculus)
6. Counterfactual simulation

### 2. Run Individual Components

#### Data Collection
```python
from src.data_collection import PriceDataCollector, NewsDataCollector

collector = PriceDataCollector(
    tickers=['RELIANCE.NS', 'TCS.NS'],
    start_date='2022-01-01',
    end_date='2024-01-01'
)
prices = collector.fetch_prices()
returns = collector.calculate_returns(prices)

news_collector = NewsDataCollector(api_key='your_key')
news = news_collector.fetch_news(query='stock market India')
```

#### Sentiment Analysis
```python
from src.sentiment_analyzer import FinBERTSentimentAnalyzer

analyzer = FinBERTSentimentAnalyzer()
sentiment = analyzer.analyze_news_df(news_df)
```

#### Causal Discovery
```python
from src.causal_discovery import CausalDiscoveryEngine

engine = CausalDiscoveryEngine(pc_alpha=0.05, tau_max=5)
results = engine.discover_causal_graph(data)
graph = results['graph']
```

#### Causal Inference
```python
from src.causal_inference import CausalInferenceEngine

inference = CausalInferenceEngine()
inference.fit(data, graph)
effect = inference.estimate_causal_effect(
    treatment='sentiment_mean',
    outcome='RELIANCE',
    intervention_value=1.5
)
print(f"Causal effect: {effect['ate']}")
```

#### Counterfactual Simulation
```python
from src.counterfactual import CounterfactualSimulator

sim = CounterfactualSimulator(graph, models, data)
result = sim.simulate_intervention(
    intervention={'sentiment_mean': 1.5},
    target_date='2023-06-15'
)
print(f"Impact: {result['causal_effect']}")
```

### 3. Using Jupyter Notebooks

The project includes 5 tutorial notebooks:

1. **01_data_collection.ipynb**: Data collection tutorial
2. **02_causal_discovery.ipynb**: Causal discovery tutorial
3. **03_causal_inference.ipynb**: Causal inference tutorial
4. **04_counterfactual.ipynb**: Counterfactual simulation tutorial
5. **05_architecture_visualization.ipynb**: System architecture visualization

Run notebooks in order for complete workflow demonstration.

---

## Algorithm Details

### PCMCI+ Algorithm

PCMCI+ (Peter-Clark Momentary Conditional Independence Plus) is a state-of-the-art algorithm for time-series causal discovery.

**Steps**:
1. **PC1 (Condition Selection)**: For each variable, select potential parents using unconditional independence tests
2. **MCI (Momentary Conditional Independence)**: Test conditional independence at specific time lags
3. **Orientation**: Orient edges using temporal information and orientation rules

**Advantages over Granger Causality**:
- Handles contemporaneous (same-time) links
- More robust to high-dimensional data
- Fewer false positives
- Explicit control of false discovery rate (pc_alpha)

**Parameters**:
- `pc_alpha`: Significance level for PC step (lower = fewer edges)
- `tau_max`: Maximum time lag to consider
- `tau_min`: Minimum time lag

### Backdoor Adjustment

The backdoor criterion is used to identify valid adjustment sets for causal effect estimation.

**Backdoor Criterion**: A set Z satisfies the backdoor criterion relative to (X, Y) if:
1. Z blocks all backdoor paths from X to Y
2. No node in Z is a descendant of X

**Estimation Formula**:
```
ATE = Σ_z [E(Y | X=1, Z=z) - E(Y | X=0, Z=z)] * P(Z=z)
```

**Implementation**:
1. Identify adjustment set using graph structure
2. Fit regression model: Y ~ X + Z
3. Predict Y under intervention X=1 for all Z
4. Predict Y under intervention X=0 for all Z
5. Average difference across all observations

### Counterfactual Propagation

Counterfactuals are computed by propagating interventions through the causal graph.

**Algorithm**:
1. Set intervened variables to intervention values
2. For each variable in topological order:
   - Skip if intervened
   - Get parent values from counterfactual state
   - Predict using fitted outcome model
   - Store prediction
3. Compare counterfactual with factual values
4. Compute impact metrics

**Impact Metrics**:
- **Causal Impact**: L2 norm of all differences
- **Most Affected Variable**: Variable with maximum absolute change
- **Individual Effects**: Difference per variable

---

## Results and Outputs

### Data Outputs

**Raw Data** (`data/raw/`):
- `prices.csv`: Daily closing prices for all tickers
- `returns.csv`: Log returns
- `news.csv`: News articles with metadata
- `sentiment_daily.csv`: Daily aggregated sentiment

**Processed Data** (`data/processed/`):
- `merged_data.csv`: Merged prices and sentiment (2665 rows × 25 columns)
- `merged_data_normalized.csv`: Z-normalized for causal discovery
- `causal_graph.graphml`: Causal graph in GraphML format
- `causal_graph.pkl`: Causal graph in Pickle format
- `causal_parents.csv`: Parent relationships for each variable

### Analysis Results

**Causal Effects** (`results/`):
- `causal_effects.csv`: Detailed causal effects with confidence intervals
- `causal_effects_all.csv`: Multiple intervention-outcome pairs
- `counterfactual_scenarios.csv`: Batch counterfactual results

**Sample Causal Effects**:
```
treatment        outcome          ate       ci_lower   ci_upper
sentiment_mean   ADANIENT         0.0070    -0.0038    0.0153
sentiment_mean   BAJFINANCE       0.0024    -0.0036    0.0079
sentiment_mean   HDFCBANK         0.0019    -0.0006    0.0039
```

### Visualizations

**Figures** (`results/figures/`):
- `causal_graph.png`: Network visualization of causal graph
- `causal_effects_heatmap.png`: Heatmap of causal effects
- `counterfactual_comparison.png`: Factual vs counterfactual bar chart
- `what_if_analysis.png`: What-if curve across intervention values

---

## Key Findings

### Causal Graph Statistics

- **Nodes**: 25 (10 stocks + 15 sentiment features)
- **Edges**: 211 directed edges
- **Average Degree**: 8.44
- **Key Drivers**: INFY (Infosys) appears as parent for many stocks
- **Sentiment Impact**: sentiment_lag features show strong causal links

### Causal Effects

**Sentiment → Stock Returns**:
- Positive sentiment causes small positive returns (ATE ≈ 0.002-0.007)
- Effect size varies by stock (ADANIENT most sensitive)
- Confidence intervals often include zero (limited statistical power)

**Correlation vs Causation**:
- Example: sentiment_mean → HDFCBANK
  - Correlation: -0.0222 (negative)
  - Causal Effect: +0.0019 (positive)
  - Difference: 0.0241 (significant!)
- Shows importance of causal inference over simple correlation

### Counterfactual Insights

**Scenario**: Sentiment = +2.0 (very positive)
- Most stocks show modest positive response
- Causal impact varies by market conditions
- Some stocks show no response (not causally connected)

---

## Comparison with Other Approaches

| Paper/Approach | Causal Discovery | Do-Calculus | Time-Series | Counterfactuals |
|----------------|------------------|-------------|-------------|-----------------|
| CausalStock    | ✓ (Custom)       | ✗           | ✓           | ✗               |
| CSHT           | Granger only     | ✗           | ✓           | ✗               |
| IRIS           | ✓ (PC)           | ✗           | ✗           | ✗               |
| CGF-LLM        | PCMCI            | ✗           | ✓           | ✗               |
| **CANDLE**     | ✓ **PCMCI+**     | ✓ **Backdoor** | ✓         | ✓               |

**Unique Features of CANDLE**:
1. Complete causal inference pipeline (discovery + inference + counterfactuals)
2. Financial sentiment integration (FinBERT)
3. Real-world stock data (NSE India)
4. Backdoor adjustment with confidence intervals
5. What-if analysis across value ranges
6. Hindsight analysis for trading decisions

---

## Limitations and Future Work

### Current Limitations

1. **DoWhy Compatibility**: Disabled due to Python 3.13 issues, using custom implementation
2. **Tigramite Dependency**: Optional, falls back to Granger causality if unavailable
3. **News API Limits**: Free tier limited to 1000 requests/day
4. **GPU Required**: FinBERT slow on CPU (fallback dummy available)
5. **Linear Models**: Counterfactuals use linear assumptions (simplified)
6. **Single Market**: Only NSE India stocks (could extend to other markets)

### Future Enhancements

1. **Better Time-Series Models**: Use VAR, LSTM, or Transformer for counterfactuals
2. **Additional Discovery Algorithms**: NOTEARS, LiNGAM, GES
3. **More Financial Datasets**: Crypto, forex, commodities
4. **Real-Time Pipeline**: Streaming data and online causal discovery
5. **Causal Reinforcement Learning**: Optimal trading with causal constraints
6. **Explainable AI**: Causal explanations for predictions
7. **Multi-Market Analysis**: Cross-market causal relationships
8. **Alternative Sentiment**: Social media, analyst reports, earnings calls

---

## Troubleshooting

### Common Issues

**Issue**: Tigramite not available
```
Solution: Falls back to Granger causality automatically
Install: pip install tigramite
```

**Issue**: FinBERT model loading fails
```
Solution: Uses dummy sentiment automatically
Install: pip install torch transformers
GPU: Ensure CUDA available for speed
```

**Issue**: NewsAPI rate limit
```
Solution: Use dummy news by setting news_api_key: null
```

**Issue**: KeyError in counterfactual visualization
```
Solution: Ensure 'intervention_value' column exists in DataFrame
Fixed in visualization.py (line 324, 343)
```

**Issue**: Missing data in merged dataset
```
Solution: Check date ranges overlap
Use ffill/bfill in merge_price_and_sentiment()
```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check data shapes:
```python
print(f"Prices: {prices.shape}")
print(f"Sentiment: {sentiment.shape}")
print(f"Merged: {merged.shape}")
```

---

## Citation

If you use CANDLE in your research, please cite:

```bibtex
@software{candle2024,
  title={CANDLE/FCIS: Causal Analysis for Financial Data},
  author={CANDLE Team},
  year={2024},
  url={https://github.com/yourusername/causal-3}
}
```

---

## License

MIT License

---

## Contact

For questions or collaborations, open an issue on GitHub.

---

## Summary

CANDLE/FCIS is a comprehensive causal inference framework for financial markets that:

1. **Collects** multi-modal data (prices + news)
2. **Analyzes** sentiment using FinBERT
3. **Discovers** causal structure using PCMCI+
4. **Estimates** causal effects using do-calculus
5. **Simulates** counterfactual scenarios

**Key Innovation**: Moves from correlation-based prediction to causal reasoning, enabling "What if?" analysis for financial decision-making.

**One Line Summary**: Move Finance AI from "What will happen?" to "Why + What if?" 🎯
