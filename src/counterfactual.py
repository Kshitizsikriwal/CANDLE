"""Counterfactual simulation module.

Third pillar of CANDLE: "What if?" scenario analysis using
counterfactual reasoning on the learned causal model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger("CANDLE")


class CounterfactualSimulator:
    """Simulate counterfactual scenarios: What if X had been different?"""
    
    def __init__(self, causal_graph, outcome_models: Dict, data: pd.DataFrame):
        """
        Initialize with learned causal structure.
        
        Args:
            causal_graph: NetworkX DiGraph with causal structure
            outcome_models: Dict mapping variable -> fitted model
            data: Historical data for reference
        """
        self.graph = causal_graph
        self.models = outcome_models
        self.data = data
        self.factual_data = data.copy()
    
    def simulate_intervention(
        self,
        intervention: Dict[str, float],
        target_date: Optional[str] = None,
        horizon: int = 1
    ) -> Dict:
        """
        Simulate: "What would have happened if we intervened on X?"
        
        Args:
            intervention: Dict mapping variable names to intervention values
            target_date: Date to simulate (default: last date in data)
            horizon: How many steps forward to simulate
        
        Returns:
            Dictionary with counterfactual outcomes
        """
        if target_date is None:
            target_idx = len(self.data) - 1
        else:
            target_idx = self.data.index.get_loc(target_date)
        
        # Get factual values at target date
        factual_values = self.data.iloc[target_idx].to_dict()
        
        # Compute counterfactual values
        counterfactual_values = self._compute_counterfactual(
            factual_values, intervention, target_idx
        )
        
        # Propagate effects forward
        cf_trajectory = [counterfactual_values]
        for h in range(1, horizon):
            next_cf = self._propagate_forward(cf_trajectory[-1], target_idx + h)
            cf_trajectory.append(next_cf)
        
        # Compare with factual
        comparison = self._compare_factual_counterfactual(
            factual_values, counterfactual_values, intervention
        )
        
        return {
            'factual': factual_values,
            'counterfactual': counterfactual_values,
            'intervention': intervention,
            'comparison': comparison,
            'causal_effect': comparison['causal_impact'],
            'trajectory': cf_trajectory if horizon > 1 else None
        }
    
    def _compute_counterfactual(
        self,
        factual: Dict[str, float],
        intervention: Dict[str, float],
        time_idx: int
    ) -> Dict[str, float]:
        """Compute counterfactual values given an intervention."""
        counterfactual = factual.copy()
        
        # Apply interventions (break incoming causal links)
        for var, value in intervention.items():
            counterfactual[var] = value
        
        # Propagate effects through causal graph
        # Process in topological order
        try:
            order = list(nx.topological_sort(self.graph))
        except:
            # If cycles exist, process by number of ancestors
            order = sorted(
                self.graph.nodes(),
                key=lambda x: len(list(self.graph.predecessors(x)))
            )
        
        for var in order:
            if var in intervention:
                continue  # Already set by intervention
            
            # Get parents
            parents = list(self.graph.predecessors(var))
            
            if not parents or var not in self.models:
                continue
            
            # Get parent values in counterfactual world
            parent_values = []
            for p in parents:
                if p in counterfactual:
                    parent_values.append(counterfactual[p])
                else:
                    parent_values.append(factual.get(p, 0))
            
            # Check for NaN values and skip prediction if found
            if any(pd.isna(v) for v in parent_values):
                # Use factual value as fallback
                counterfactual[var] = factual.get(var, 0)
                continue
            
            # Predict using outcome model
            if self.models[var] is not None:
                X = np.array(parent_values).reshape(1, -1)
                pred = self.models[var].predict(X)[0]
                counterfactual[var] = pred
        
        return counterfactual
    
    def _propagate_forward(
        self,
        current_values: Dict[str, float],
        next_idx: int
    ) -> Dict[str, float]:
        """Propagate counterfactual state to next time step."""
        # In time series, we need to consider lagged effects
        # This is a simplified version - full version would use time-aware models
        return current_values
    
    def _compare_factual_counterfactual(
        self,
        factual: Dict[str, float],
        counterfactual: Dict[str, float],
        intervention: Dict[str, float]
    ) -> Dict:
        """Compare factual and counterfactual outcomes."""
        
        # Calculate differences for all variables
        differences = {}
        for var in factual:
            if var in counterfactual:
                differences[var] = counterfactual[var] - factual[var]
        
        # Find most affected variables (excluding intervened ones)
        affected_vars = {k: v for k, v in differences.items() if k not in intervention}
        
        if affected_vars:
            most_affected = max(affected_vars.items(), key=lambda x: abs(x[1]))
        else:
            most_affected = (None, 0)
        
        # Calculate causal impact score (L2 norm of differences)
        causal_impact = np.sqrt(sum([d**2 for d in differences.values()]))
        
        return {
            'differences': differences,
            'most_affected_variable': most_affected[0],
            'most_affected_change': most_affected[1],
            'causal_impact': causal_impact,
            'intervened_variables': list(intervention.keys())
        }
    
    def what_if_analysis(
        self,
        variable: str,
        value_range: Tuple[float, float],
        n_scenarios: int = 10,
        outcome_vars: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Analyze: "What would happen if variable X took different values?"
        
        Returns DataFrame with scenarios and their predicted outcomes.
        """
        if outcome_vars is None:
            outcome_vars = [n for n in self.graph.nodes() if n != variable]
        
        scenarios = []
        values = np.linspace(value_range[0], value_range[1], n_scenarios)
        
        for val in values:
            result = self.simulate_intervention({variable: val})
            
            scenario = {
                'intervention_variable': variable,
                'intervention_value': val
            }
            
            # Add factual values
            for var in outcome_vars:
                if var in result['factual']:
                    scenario[f'{var}_factual'] = result['factual'][var]
            
            # Add counterfactual values
            for var in outcome_vars:
                if var in result['counterfactual']:
                    scenario[f'{var}_counterfactual'] = result['counterfactual'][var]
                    scenario[f'{var}_difference'] = (
                        result['counterfactual'][var] - result['factual'][var]
                    )
            
            scenario['total_impact'] = result['comparison']['causal_impact']
            scenarios.append(scenario)
        
        return pd.DataFrame(scenarios)
    
    def hindsight_analysis(
        self,
        target_var: str,
        target_date: str,
        alternative_value: float
    ) -> Dict:
        """
        Analyze: "What if we had acted differently on a specific date?"
        
        Useful for evaluating past trading decisions.
        """
        # Get factual outcome
        if target_date in self.data.index:
            factual = self.data.loc[target_date, target_var]
        else:
            raise ValueError(f"Date {target_date} not found in data")
        
        # Simulate counterfactual
        result = self.simulate_intervention(
            {target_var: alternative_value},
            target_date=target_date
        )
        
        counterfactual_outcomes = {
            k: v for k, v in result['counterfactual'].items()
            if k != target_var
        }
        
        # Calculate what the return would have been
        # Assuming we're analyzing stock returns
        return_diff = counterfactual_outcomes.get('return', 0) - self.data.loc[target_date].get('return', 0)
        
        return {
            'date': target_date,
            'variable': target_var,
            'factual_value': factual,
            'alternative_value': alternative_value,
            'counterfactual_returns': counterfactual_outcomes,
            'return_difference': return_diff,
            'better_outcome': return_diff > 0
        }
    
    def batch_counterfactuals(
        self,
        interventions_list: List[Dict[str, float]]
    ) -> pd.DataFrame:
        """Run multiple counterfactual scenarios and compare."""
        results = []
        
        for i, intervention in enumerate(interventions_list):
            try:
                result = self.simulate_intervention(intervention)
                
                row = {
                    'scenario_id': i,
                    'intervention': str(intervention),
                    'causal_impact': result['comparison']['causal_impact'],
                    'most_affected': result['comparison']['most_affected_variable'],
                    'most_affected_change': result['comparison']['most_affected_change']
                }
                
                # Add key counterfactual values
                for var, val in result['counterfactual'].items():
                    row[f'cf_{var}'] = val
                
                results.append(row)
                
            except Exception as e:
                logger.warning(f"Scenario {i} failed: {e}")
        
        return pd.DataFrame(results)


# Import networkx at module level for type hints
import networkx as nx
