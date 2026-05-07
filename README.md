# CANDLE / FCIS
## Financial Causal Intelligence System
### From Correlation to Causation in Financial AI

<p align="center">
  <img src="/src/assets/architecture.png" width="100%">
</p>

---

## Overview

CANDLE (Causal And News-Driven LLM Engine) is a research-oriented financial causal intelligence framework designed to move beyond correlation-based prediction toward intervention-aware and counterfactual financial reasoning.

Unlike conventional quantitative systems that rely purely on statistical associations, CANDLE integrates:

- Temporal causal discovery
- Structural causal inference
- Counterfactual simulation
- Multimodal financial reasoning
- Federated causal learning

to generate explainable and causally grounded alpha signals.

The framework combines market data, macroeconomic signals, technical indicators, and financial news embeddings into a unified temporal causal graph capable of answering:

- What caused a market movement?
- What would happen under intervention?
- What would have happened if an event never occurred?

---

# Motivation

Modern financial AI systems suffer from three major limitations:

1. Spurious correlations
2. Regime instability
3. Lack of causal explainability

Traditional ML answers:

> "What is likely to happen?"

CANDLE attempts to answer:

> "Why did it happen?"  
> "What if conditions changed?"  
> "What would have happened otherwise?"

This framework is heavily inspired by Judea Pearl’s Structural Causal Models (SCMs) and the Ladder of Causation.

---

# Core Research Contributions

## 1. Joint Temporal-Multimodal Causal Graph

Unlike late-fusion pipelines, CANDLE models:

- price variables,
- technical indicators,
- macroeconomic variables,
- and FinBERT-encoded news embeddings

inside a unified causal graph.

---

## 2. Structural Causal Inference Layer

The framework performs intervention-based estimation using:

\[
P(Y \mid do(X=x))
\]

instead of standard observational probability:

\[
P(Y \mid X=x)
\]

using Pearl’s backdoor adjustment criterion.

---

## 3. Counterfactual Financial Simulation

CANDLE supports financial what-if reasoning:

- What if earnings had not missed expectations?
- What if interest rates were unchanged?
- What if geopolitical news never occurred?

Counterfactual propagation estimates alternative return trajectories under hypothetical interventions.

---

## 4. Federated Causal Discovery

A privacy-preserving layer enables institutions to share:

- causal graph parameters,
- edge weights,
- structural patterns

without exposing proprietary raw market data.

---

# System Architecture

The framework contains eight major layers:

1. Data Ingestion Layer
2. Preprocessing & Feature Engineering
3. Temporal Causal Discovery Engine
4. Causal Inference Layer
5. Counterfactual Simulation Engine
6. Alpha Generation Layer
7. Visualization & API Layer
8. Federated Privacy Layer

---

# Pipeline

## Step 1 — Data Collection

Sources include:

- Yahoo Finance
- NSE/BSE
- SEC Filings
- NewsAPI
- Reuters
- Bloomberg
- Social media feeds

---

## Step 2 — NLP & Sentiment Encoding

Financial news is processed using:

- FinBERT
- HuggingFace Transformers
- Sentence embeddings

Outputs include:

- sentiment polarity
- semantic embeddings
- event representations

---

## Step 3 — Temporal Causal Discovery

Algorithms explored:

- PCMCI+
- NOTEARS
- LiNGAM
- DAG-based optimization

Outputs:

- lagged causal edges
- contemporaneous causal links
- weighted directed acyclic graphs

---

## Step 4 — Causal Inference

The framework estimates:

- Average Treatment Effect (ATE)
- Conditional Average Treatment Effect (CATE)
- Interventional expectations

using:

- backdoor adjustment
- do-calculus
- structural causal models

---

## Step 5 — Counterfactual Engine

The simulation engine propagates interventions through the learned DAG to estimate alternate financial trajectories.

---

# Example Financial Counterfactual

Observed reality:

> Fed raises interest rates  
> JPMorgan falls -1.8%

Counterfactual query:

> What if rates had not increased?

Estimated counterfactual outcome:

> JPMorgan return = +0.3%

Estimated causal effect:

\[
-2.1\%
\]

---

# Technology Stack

| Category | Tools |
|---|---|
| Data | yfinance, pandas, numpy |
| NLP | FinBERT, HuggingFace |
| Causal Discovery | tigramite, NOTEARS |
| Inference | DoWhy, EconML |
| ML/DL | PyTorch, XGBoost |
| Visualization | Plotly, matplotlib |
| Databases | PostgreSQL, Neo4j |
| Infra | Docker, Kubernetes |
| Experiment Tracking | MLflow |

---

# Repository Structure

```text
src/
├── ingestion/
├── preprocessing/
├── embeddings/
├── causal_discovery/
├── causal_inference/
├── counterfactuals/
├── alpha_generation/
├── federated/
└── visualization/
