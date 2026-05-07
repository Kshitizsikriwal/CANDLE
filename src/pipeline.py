"""Main pipeline orchestrator for CANDLE/FCIS framework.

Integrates all three pillars:
1. Causal Discovery (PCMCI+)
2. Causal Inference (Do-calculus)
3. Counterfactual Simulation
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

from utils import load_config, setup_logging, ensure_directories, z_normalize, clean_ticker_names
from data_collection import PriceDataCollector, NewsDataCollector, merge_price_and_sentiment
from sentiment_analyzer import FinBERTSentimentAnalyzer, create_sentiment_features
from causal_discovery import CausalDiscoveryEngine
from causal_inference import CausalInferenceEngine
from counterfactual import CounterfactualSimulator

logger = logging.getLogger("CANDLE")


class CANDLEPipeline:
    """Main pipeline for Causal Analysis in Finance."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging(
            self.config['output']['logs_path'],
            "INFO"
        )
        ensure_directories(self.config)
        
        # Initialize components
        self.price_collector = None
        self.news_collector = None
        self.sentiment_analyzer = None
        self.causal_discovery = None
        self.causal_inference = None
        self.counterfactual_sim = None
        
        # Data storage
        self.prices = None
        self.returns = None
        self.news = None
        self.sentiment = None
        self.merged_data = None
        self.causal_graph = None
        self.discovery_results = None
        
    def run_data_collection(self) -> pd.DataFrame:
        """Step 1: Collect price and news data."""
        logger.info("=" * 60)
        logger.info("STEP 1: Data Collection")
        logger.info("=" * 60)
        
        # Collect prices
        self.price_collector = PriceDataCollector(
            tickers=self.config['data']['tickers'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        self.prices = self.price_collector.fetch_prices()
        self.returns = self.price_collector.calculate_returns(self.prices)
        
        # Save raw prices
        raw_path = self.config['data']['raw_data_path']
        self.prices.to_csv(f"{raw_path}/prices.csv")
        self.returns.to_csv(f"{raw_path}/returns.csv")
        logger.info(f"Saved prices to {raw_path}/prices.csv")
        
        # Collect news (or use dummy) - pass date range for alignment
        self.news_collector = NewsDataCollector(
            api_key=self.config['data']['news_api_key'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        self.news = self.news_collector.fetch_news(
            query="stock market India OR NSE OR finance OR economy",
            from_date=self.config['data']['start_date'],
            to_date=self.config['data']['end_date'],
            max_results=1000
        )
        
        self.news.to_csv(f"{raw_path}/news.csv", index=False)
        logger.info(f"Saved news to {raw_path}/news.csv")
        
        return self.returns
    
    def run_sentiment_analysis(self) -> pd.DataFrame:
        """Step 2: Analyze sentiment using FinBERT."""
        logger.info("=" * 60)
        logger.info("STEP 2: Sentiment Analysis")
        logger.info("=" * 60)
        
        if self.news is None:
            raise ValueError("Run data_collection first")
        
        # Initialize FinBERT
        self.sentiment_analyzer = FinBERTSentimentAnalyzer(
            model_name=self.config['sentiment']['model_name']
        )
        
        # Analyze sentiment
        self.sentiment = self.sentiment_analyzer.analyze_news_df(
            self.news, 
            text_column='title'
        )
        
        # Create additional features
        self.sentiment = create_sentiment_features(self.sentiment)
        
        # Save
        raw_path = self.config['data']['raw_data_path']
        self.sentiment.to_csv(f"{raw_path}/sentiment_daily.csv")
        logger.info(f"Saved sentiment to {raw_path}/sentiment_daily.csv")
        
        return self.sentiment
    
    def merge_and_preprocess(self) -> pd.DataFrame:
        """Step 3: Merge price and sentiment data."""
        logger.info("=" * 60)
        logger.info("STEP 3: Data Merging & Preprocessing")
        logger.info("=" * 60)
        
        if self.returns is None or self.sentiment is None:
            raise ValueError("Run data_collection and sentiment_analysis first")
        
        # Merge with logging for debugging
        self.merged_data = merge_price_and_sentiment(
            self.returns, self.sentiment, logger=logger
        )
        
        # Z-normalize for causal discovery
        self.merged_data_norm = z_normalize(self.merged_data)
        
        # Save
        processed_path = self.config['data']['processed_data_path']
        self.merged_data.to_csv(f"{processed_path}/merged_data.csv")
        self.merged_data_norm.to_csv(f"{processed_path}/merged_data_normalized.csv")
        
        logger.info(f"Merged data shape: {self.merged_data.shape}")
        logger.info(f"Date range: {self.merged_data.index[0]} to {self.merged_data.index[-1]}")
        logger.info(f"Variables: {list(self.merged_data.columns)}")
        
        return self.merged_data
    
    def run_causal_discovery(self) -> Dict:
        """Step 4: Discover causal graph using PCMCI+."""
        logger.info("=" * 60)
        logger.info("STEP 4: Causal Discovery (PCMCI+)")
        logger.info("=" * 60)
        
        if self.merged_data_norm is None:
            raise ValueError("Run merge_and_preprocess first")
        
        # Initialize causal discovery engine
        self.causal_discovery = CausalDiscoveryEngine(
            pc_alpha=self.config['causal_discovery']['pc_alpha'],
            tau_max=self.config['causal_discovery']['tau_max'],
            tau_min=self.config['causal_discovery']['tau_min']
        )
        
        # Run discovery
        self.discovery_results = self.causal_discovery.discover_causal_graph(
            self.merged_data_norm
        )
        
        self.causal_graph = self.discovery_results['graph']
        
        # Log results
        logger.info("Causal Graph Discovered:")
        logger.info(f"Nodes: {list(self.causal_graph.nodes())}")
        logger.info(f"Edges: {list(self.causal_graph.edges(data=True))}")
        
        # Save results
        processed_path = self.config['data']['processed_data_path']
        
        # Save parents dict
        parents_df = pd.DataFrame([
            {'variable': k, 'parents': str(v)}
            for k, v in self.discovery_results['parents'].items()
        ])
        parents_df.to_csv(f"{processed_path}/causal_parents.csv", index=False)
        
        return self.discovery_results
    
    def run_causal_inference(self) -> pd.DataFrame:
        """Step 5: Estimate causal effects using do-calculus."""
        logger.info("=" * 60)
        logger.info("STEP 5: Causal Inference (Do-Calculus)")
        logger.info("=" * 60)
        
        if self.causal_graph is None or self.merged_data is None:
            raise ValueError("Run causal_discovery first")
        
        # Initialize inference engine
        self.causal_inference = CausalInferenceEngine(
            method=self.config['inference']['method'],
            confidence_level=self.config['inference']['confidence_level']
        )
        
        # Fit models
        self.causal_inference.fit(
            self.merged_data,
            self.causal_graph
        )
        
        # Define interventions to test
        stock_cols = [c for c in self.merged_data.columns 
                      if c not in ['sentiment_mean', 'sentiment_std', 'news_count']]
        sentiment_cols = [c for c in self.merged_data.columns if 'sentiment' in c]
        
        interventions = []
        
        # Test sentiment -> stock effects
        for sent_col in sentiment_cols[:2]:  # Limit to first 2 sentiment features
            interventions.append((sent_col, 1.0))  # High positive sentiment
            interventions.append((sent_col, -1.0))  # High negative sentiment
        
        # Test stock -> stock effects
        for stock in stock_cols[:3]:  # Limit to first 3 stocks
            interventions.append((stock, 0.02))  # 2% return
            interventions.append((stock, -0.02))  # -2% return
        
        # Estimate all effects
        effects_df = self.causal_inference.estimate_all_effects(
            interventions,
            stock_cols[:5]  # Outcome variables
        )
        
        # Save results
        results_path = self.config['output']['results_path']
        effects_df.to_csv(f"{results_path}/causal_effects.csv", index=False)
        
        logger.info(f"Estimated {len(effects_df)} causal effects")
        
        if len(effects_df) > 0 and 'ate' in effects_df.columns:
            logger.info("\nTop 5 strongest effects:")
            top_effects = effects_df.nlargest(min(5, len(effects_df)), 'ate')[['treatment', 'outcome', 'ate']]
            for _, row in top_effects.iterrows():
                logger.info(f"  {row['treatment']} -> {row['outcome']}: {row['ate']:.4f}")
        else:
            logger.warning("No causal effects could be estimated (this can happen with fallback mode)")
        
        return effects_df
    
    def run_counterfactual_simulation(self) -> pd.DataFrame:
        """Step 6: Run counterfactual scenarios."""
        logger.info("=" * 60)
        logger.info("STEP 6: Counterfactual Simulation")
        logger.info("=" * 60)
        
        if self.causal_inference is None or self.causal_graph is None:
            raise ValueError("Run causal_inference first")
        
        # Initialize counterfactual simulator
        self.counterfactual_sim = CounterfactualSimulator(
            causal_graph=self.causal_graph,
            outcome_models=self.causal_inference.models,
            data=self.merged_data
        )
        
        # Define scenarios
        stock_cols = [c for c in self.merged_data.columns 
                      if c not in ['sentiment_mean', 'sentiment_std', 'news_count']]
        
        scenarios = []
        
        # Scenario 1: What if sentiment was very positive?
        if 'sentiment_mean' in self.merged_data.columns:
            scenarios.append({'sentiment_mean': 1.5})
        
        # Scenario 2: What if a major stock crashed?
        if stock_cols:
            scenarios.append({stock_cols[0]: -0.05})  # -5% return
        
        # Scenario 3: What if sentiment and market were both down?
        if stock_cols and 'sentiment_mean' in self.merged_data.columns:
            scenarios.append({
                'sentiment_mean': -1.5,
                stock_cols[0]: -0.03
            })
        
        # Run batch counterfactuals
        if scenarios:
            cf_results = self.counterfactual_sim.batch_counterfactuals(scenarios)
            
            # Save
            results_path = self.config['output']['results_path']
            cf_results.to_csv(f"{results_path}/counterfactual_scenarios.csv", index=False)
            
            logger.info(f"Ran {len(scenarios)} counterfactual scenarios")
            logger.info("\nResults:")
            for _, row in cf_results.iterrows():
                logger.info(f"  Scenario {row['scenario_id']}: Impact = {row['causal_impact']:.4f}")
            
            return cf_results
        
        return pd.DataFrame()
    
    def run_full_pipeline(self) -> Dict:
        """Run complete CANDLE pipeline."""
        logger.info("\n" + "=" * 60)
        logger.info("CANDLE/FCIS - Full Pipeline Execution")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Execute all steps
        self.run_data_collection()
        self.run_sentiment_analysis()
        self.merge_and_preprocess()
        self.run_causal_discovery()
        self.run_causal_inference()
        self.run_counterfactual_simulation()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete!")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info("=" * 60)
        
        # Summary
        summary = {
            'duration_seconds': duration,
            'data_points': len(self.merged_data) if self.merged_data is not None else 0,
            'n_variables': len(self.merged_data.columns) if self.merged_data is not None else 0,
            'n_causal_edges': len(self.causal_graph.edges()) if self.causal_graph is not None else 0,
            'output_paths': {
                'prices': f"{self.config['data']['raw_data_path']}/prices.csv",
                'sentiment': f"{self.config['data']['raw_data_path']}/sentiment_daily.csv",
                'merged_data': f"{self.config['data']['processed_data_path']}/merged_data.csv",
                'causal_effects': f"{self.config['output']['results_path']}/causal_effects.csv",
                'counterfactuals': f"{self.config['output']['results_path']}/counterfactual_scenarios.csv"
            }
        }
        
        logger.info("\nSummary:")
        for key, value in summary.items():
            if key != 'output_paths':
                logger.info(f"  {key}: {value}")
        
        return summary


def main():
    """Main entry point."""
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    pipeline = CANDLEPipeline(config_path)
    summary = pipeline.run_full_pipeline()
    
    print("\n" + "=" * 60)
    print("CANDLE Execution Complete!")
    print("=" * 60)
    print(f"Data points processed: {summary['data_points']}")
    print(f"Variables analyzed: {summary['n_variables']}")
    print(f"Causal edges discovered: {summary['n_causal_edges']}")
    print("\nOutput files:")
    for name, path in summary['output_paths'].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
