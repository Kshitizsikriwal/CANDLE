"""Causal Discovery module using PCMCI+ algorithm from tigramite.

Implements the first pillar of CANDLE: building causal graphs from
time-series data (prices + sentiment).
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("CANDLE")

# Import tigramite components
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.independence_tests.gpdc import GPDC
    from tigramite.independence_tests.cmiknn import CMIknn
    TIGRAMITE_AVAILABLE = True
    logger.info("Tigramite imported successfully")
except ImportError as e:
    TIGRAMITE_AVAILABLE = False
    logger.warning(f"Tigramite not available: {e}. Using fallback implementation.")


class CausalDiscoveryEngine:
    """Causal discovery using PCMCI+ algorithm."""
    
    def __init__(
        self,
        pc_alpha: float = 0.05,
        tau_max: int = 5,
        tau_min: int = 1,
        independence_test: str = "parcorr"
    ):
        self.pc_alpha = pc_alpha
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.independence_test = independence_test
        self.graph = None
        self.var_names = None
        
        if not TIGRAMITE_AVAILABLE:
            logger.warning("Running in fallback mode with simplified algorithm")
    
    def discover_causal_graph(
        self, 
        data: pd.DataFrame,
        var_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Discover causal graph from time series data.
        
        Args:
            data: Time series data (samples x variables)
            var_names: Variable names (defaults to column names)
        
        Returns:
            Dictionary containing graph, parents, and results
        """
        self.var_names = var_names or list(data.columns)
        
        if TIGRAMITE_AVAILABLE:
            return self._discover_with_pcmci(data)
        else:
            return self._discover_fallback(data)
    
    def _discover_with_pcmci(self, data: pd.DataFrame) -> Dict:
        """Use tigramite PCMCI+ for causal discovery."""
        logger.info("Running PCMCI+ causal discovery...")
        
        # Convert to numpy array
        data_array = data.values
        T, N = data_array.shape
        
        # Create tigramite DataFrame
        dataframe = pp.DataFrame(
            data_array,
            datatime=np.arange(T),
            var_names=self.var_names
        )
        
        # Select independence test
        if self.independence_test == "parcorr":
            cond_ind_test = ParCorr()
        elif self.independence_test == "gpdc":
            cond_ind_test = GPDC()
        elif self.independence_test == "cmiknn":
            cond_ind_test = CMIknn()
        else:
            cond_ind_test = ParCorr()
        
        # Run PCMCI+ algorithm
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=1
        )
        
        # Run PCMCI+ with selected parameters
        results = pcmci.run_pcmciplus(
            tau_min=self.tau_min,
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha
        )
        
        # Extract graph structure
        graph = results['graph']
        val_matrix = results['val_matrix']
        p_matrix = results['p_matrix']
        
        # Convert to networkx for easier manipulation
        nx_graph = self._tigramite_to_networkx(graph, val_matrix)
        
        # Get causal parents for each variable
        parents = self._extract_parents(graph)
        
        self.graph = nx_graph
        
        logger.info(f"Causal discovery complete. Found {len(parents)} causal relationships")
        
        return {
            'graph': nx_graph,
            'tigramite_graph': graph,
            'val_matrix': val_matrix,
            'p_matrix': p_matrix,
            'parents': parents,
            'results': results
        }
    
    def _discover_fallback(self, data: pd.DataFrame) -> Dict:
        """Fallback implementation using Granger causality and correlation."""
        logger.info("Running fallback causal discovery (Granger causality)...")
        
        from statsmodels.tsa.stattools import grangercausalitytests
        
        n_vars = len(data.columns)
        graph = np.zeros((n_vars, n_vars, self.tau_max + 1), dtype=bool)
        val_matrix = np.zeros((n_vars, n_vars, self.tau_max + 1))
        
        # Test Granger causality between each pair
        for i, target in enumerate(data.columns):
            for j, cause in enumerate(data.columns):
                if i == j:
                    continue
                
                try:
                    # Prepare data for Granger test
                    test_data = data[[cause, target]].dropna()
                    
                    if len(test_data) < 30:
                        continue
                    
                    # Run Granger causality test
                    gc_result = grangercausalitytests(
                        test_data, 
                        maxlag=self.tau_max,
                        verbose=False
                    )
                    
                    # Check if significant at any lag
                    for lag in range(1, self.tau_max + 1):
                        p_value = gc_result[lag][0]['ssr_ftest'][1]
                        if p_value < self.pc_alpha:
                            graph[j, i, lag] = True
                            val_matrix[j, i, lag] = 1 - p_value
                            
                except Exception as e:
                    logger.debug(f"Granger test failed for {cause} -> {target}: {e}")
        
        # Convert to networkx
        nx_graph = self._tigramite_to_networkx(graph, val_matrix)
        parents = self._extract_parents(graph)
        
        self.graph = nx_graph
        
        return {
            'graph': nx_graph,
            'tigramite_graph': graph,
            'val_matrix': val_matrix,
            'p_matrix': None,
            'parents': parents,
            'results': None
        }
    
    def _tigramite_to_networkx(
        self, 
        graph: np.ndarray, 
        val_matrix: np.ndarray
    ) -> nx.DiGraph:
        """Convert tigramite graph to networkx DiGraph."""
        G = nx.DiGraph()
        
        n_vars = len(self.var_names)
        
        # Add nodes
        for var in self.var_names:
            G.add_node(var)
        
        # Add edges (only contemporaneous and lagged directed)
        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(self.tau_max + 1):
                    if graph[i, j, tau]:
                        cause = self.var_names[i]
                        effect = self.var_names[j]
                        weight = abs(val_matrix[i, j, tau]) if val_matrix is not None else 1.0
                        
                        if tau == 0:
                            # Contemporaneous link
                            G.add_edge(cause, effect, weight=weight, lag=0, type='contemp')
                        else:
                            # Lagged link
                            G.add_edge(cause, effect, weight=weight, lag=tau, type='lagged')
        
        return G
    
    def _extract_parents(self, graph: np.ndarray) -> Dict[str, List[Tuple[str, int]]]:
        """Extract causal parents for each variable."""
        parents = {}
        
        for j, target in enumerate(self.var_names):
            target_parents = []
            
            for i, cause in enumerate(self.var_names):
                for tau in range(self.tau_max + 1):
                    if graph[i, j, tau]:
                        target_parents.append((cause, tau))
            
            parents[target] = target_parents
        
        return parents
    
    def get_causal_order(self) -> List[str]:
        """Get topological causal ordering of variables."""
        if self.graph is None:
            raise ValueError("Run discover_causal_graph first")
        
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            logger.warning("Graph contains cycles, returning degree-based order")
            # Return by in-degree
            degrees = dict(self.graph.in_degree())
            return sorted(degrees.keys(), key=lambda x: degrees[x])
    
    def get_markov_blanket(self, target: str) -> List[str]:
        """Get Markov blanket of target variable (parents + children + parents of children)."""
        if self.graph is None:
            raise ValueError("Run discover_causal_graph first")
        
        parents = list(self.graph.predecessors(target))
        children = list(self.graph.successors(target))
        parents_of_children = []
        
        for child in children:
            parents_of_children.extend(self.graph.predecessors(child))
        
        # Remove target itself and duplicates
        blanket = list(set(parents + children + parents_of_children) - {target})
        
        return blanket
