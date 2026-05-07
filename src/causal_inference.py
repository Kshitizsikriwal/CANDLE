"""Causal Inference module implementing Pearl's do-calculus.

Second pillar of CANDLE: Structural inference using backdoor criterion,
front-door criterion, and do-calculus for estimating causal effects.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger("CANDLE")

# DoWhy disabled due to Python 3.13 compatibility issues
# Always use custom implementation which is more reliable
DOWHY_AVAILABLE = False
logger.info("DoWhy disabled. Using custom causal inference implementation.")


class CausalInferenceEngine:
    """Causal inference using Pearl's do-calculus."""
    
    def __init__(self, method: str = "backdoor", confidence_level: float = 0.95):
        self.method = method
        self.confidence_level = confidence_level
        self.graph = None
        self.data = None
        self.models = {}
    
    def fit(
        self, 
        data: pd.DataFrame, 
        graph: nx.DiGraph,
        outcome_model: str = "ridge"
    ):
        """Fit the causal inference engine with graph and data."""
        self.data = data
        self.graph = graph
        
        # Fit models for each variable given its parents
        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            if parents:
                self.models[node] = self._fit_outcome_model(
                    node, parents, outcome_model
                )
    
    def estimate_causal_effect(
        self,
        treatment: str,
        outcome: str,
        intervention_value: float,
        control_value: float = 0.0,
        adjustment_set: Optional[List[str]] = None
    ) -> Dict:
        """
        Estimate causal effect P(Y | do(X=x)) using do-calculus.
        
        Args:
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            intervention_value: Value to set treatment to
            control_value: Baseline value for comparison
            adjustment_set: Variables to adjust for (if None, auto-detect)
        
        Returns:
            Dictionary with causal effect estimate and confidence interval
        """
        if adjustment_set is None:
            adjustment_set = self._find_adjustment_set(treatment, outcome)
        
        logger.info(f"Estimating effect of {treatment} on {outcome}")
        logger.info(f"Adjustment set: {adjustment_set}")
        
        # Use DoWhy if available
        if DOWHY_AVAILABLE:
            return self._estimate_with_dowhy(
                treatment, outcome, intervention_value, control_value, adjustment_set
            )
        else:
            return self._estimate_custom(
                treatment, outcome, intervention_value, control_value, adjustment_set
            )
    
    def _estimate_with_dowhy(
        self,
        treatment: str,
        outcome: str,
        intervention_value: float,
        control_value: float,
        adjustment_set: List[str]
    ) -> Dict:
        """Skip DoWhy due to Python 3.13 compatibility issues - use custom instead."""
        # DoWhy has bytes/string encoding issues with Python 3.13
        # Fall back to custom implementation which is more reliable
        return self._estimate_custom(
            treatment, outcome, intervention_value, control_value, adjustment_set
        )
    
    def _estimate_custom(
        self,
        treatment: str,
        outcome: str,
        intervention_value: float,
        control_value: float,
        adjustment_set: List[str]
    ) -> Dict:
        """Custom causal effect estimation using backdoor adjustment."""
        
        # Prepare data
        df = self.data[[outcome, treatment] + adjustment_set].dropna()
        
        if len(df) < 10:
            raise ValueError("Insufficient data for estimation")
        
        # Fit outcome model: E[Y | X, Z]
        features = [treatment] + adjustment_set
        X = df[features].values
        y = df[outcome].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        
        # Estimate E[Y | do(X=x)] by averaging over Z
        Z_samples = df[adjustment_set].values
        
        # Scenario 1: X = intervention_value
        X_intervened = np.column_stack([
            np.full(len(Z_samples), intervention_value),
            Z_samples
        ])
        X_intervened_scaled = scaler.transform(X_intervened)
        y_pred_intervened = model.predict(X_intervened_scaled)
        
        # Scenario 2: X = control_value
        X_control = np.column_stack([
            np.full(len(Z_samples), control_value),
            Z_samples
        ])
        X_control_scaled = scaler.transform(X_control)
        y_pred_control = model.predict(X_control_scaled)
        
        # Average treatment effect
        ate = np.mean(y_pred_intervened - y_pred_control)
        
        # Confidence interval via bootstrap
        bootstrap_ates = []
        n_bootstrap = 100
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(df), size=len(df), replace=True)
            X_boot = X_scaled[indices]
            y_boot = y[indices]
            
            model_boot = Ridge(alpha=1.0)
            model_boot.fit(X_boot, y_boot)
            
            y_int_boot = model_boot.predict(X_intervened_scaled[indices])
            y_ctrl_boot = model_boot.predict(X_control_scaled[indices])
            bootstrap_ates.append(np.mean(y_int_boot - y_ctrl_boot))
        
        ci_lower = np.percentile(bootstrap_ates, (1 - self.confidence_level) * 50)
        ci_upper = np.percentile(bootstrap_ates, 100 - (1 - self.confidence_level) * 50)
        
        return {
            'treatment': treatment,
            'outcome': outcome,
            'ate': ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'intervention_value': intervention_value,
            'control_value': control_value,
            'causal_effect': ate,
            'adjustment_set': adjustment_set,
            'method': 'custom_backdoor',
            'individual_effects': y_pred_intervened - y_pred_control
        }
    
    def _find_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """Find valid adjustment set using backdoor criterion."""
        # Get all nodes
        nodes = list(self.graph.nodes())
        
        # Simple approach: use Markov blanket of treatment minus outcome
        parents = list(self.graph.predecessors(treatment))
        children = list(self.graph.successors(treatment))
        
        # Start with parents (direct confounders)
        adjustment_set = parents
        
        # Add parents of children (if not treatment or outcome)
        for child in children:
            for parent in self.graph.predecessors(child):
                if parent != treatment and parent != outcome and parent not in adjustment_set:
                    adjustment_set.append(parent)
        
        # Ensure outcome is not in adjustment set
        adjustment_set = [x for x in adjustment_set if x != outcome]
        
        return adjustment_set
    
    def _fit_outcome_model(
        self, 
        target: str, 
        parents: List[str], 
        model_type: str = "ridge"
    ):
        """Fit outcome model for a variable given its parents."""
        df = self.data[[target] + parents].dropna()
        
        if len(df) < 10:
            return None
        
        X = df[parents].values
        y = df[target].values
        
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "lasso":
            model = Lasso(alpha=0.1)
        elif model_type == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gb":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            model = Ridge(alpha=1.0)
        
        model.fit(X, y)
        return model
    
    def _estimate_ite(
        self,
        treatment: str,
        outcome: str,
        intervention_value: float,
        control_value: float,
        adjustment_set: List[str]
    ) -> np.ndarray:
        """Estimate Individual Treatment Effects."""
        df = self.data[[outcome, treatment] + adjustment_set].dropna()
        
        X = df[[treatment] + adjustment_set].values
        y = df[outcome].values
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        # Predict under intervention
        X_int = X.copy()
        X_int[:, 0] = intervention_value
        y_int = model.predict(X_int)
        
        # Predict under control
        X_ctrl = X.copy()
        X_ctrl[:, 0] = control_value
        y_ctrl = model.predict(X_ctrl)
        
        return y_int - y_ctrl
    
    def _graph_to_gml(self) -> str:
        """Convert networkx graph to GML string for DoWhy."""
        import io
        
        # Write to string buffer
        buffer = io.StringIO()
        nx.write_gml(self.graph, buffer)
        return buffer.getvalue()
    
    def estimate_all_effects(
        self,
        interventions: List[Tuple[str, float]],
        outcomes: List[str]
    ) -> pd.DataFrame:
        """Estimate causal effects for multiple interventions and outcomes."""
        results = []
        
        for treatment, value in interventions:
            for outcome in outcomes:
                if treatment != outcome:
                    try:
                        effect = self.estimate_causal_effect(
                            treatment, outcome, value
                        )
                        results.append({
                            'treatment': treatment,
                            'outcome': outcome,
                            'intervention_value': value,
                            'ate': effect['ate'],
                            'ci_lower': effect.get('ci_lower', np.nan),
                            'ci_upper': effect.get('ci_upper', np.nan),
                            'method': effect['method']
                        })
                    except Exception as e:
                        logger.warning(f"Failed to estimate {treatment} -> {outcome}: {e}")
        
        return pd.DataFrame(results)
