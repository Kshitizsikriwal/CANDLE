"""Advanced architecture visualization for CANDLE/FCIS.

Generates publication-quality DAG visualizations matching the framework architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ArchitectureVisualizer:
    """Create framework architecture diagrams like the reference image."""
    
    def __init__(self, figsize=(24, 16)):
        self.figsize = figsize
        self.colors = {
            'data_layer': '#E3F2FD',      # Light blue
            'preprocess': '#E8F5E9',       # Light green
            'discovery': '#F3E5F5',        # Light purple
            'inference': '#FFEBEE',        # Light red
            'counterfactual': '#FFF3E0',   # Light orange
            'alpha': '#E0F2F1',            # Light teal
            'output': '#FBE9E7',           # Light coral
            'border_data': '#1976D2',
            'border_preprocess': '#388E3C',
            'border_discovery': '#7B1FA2',
            'border_inference': '#D32F2F',
            'border_counter': '#F57C00',
            'border_alpha': '#00796B',
            'border_output': '#E64A19',
            'node_news': '#90CAF9',
            'node_macro': '#A5D6A7',
            'node_stock': '#CE93D8',
            'node_technical': '#FFCC80',
            'edge_positive': '#4CAF50',
            'edge_negative': '#F44336',
            'edge_neutral': '#9E9E9E'
        }
    
    def create_framework_architecture(
        self,
        causal_graph: nx.DiGraph,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create full framework architecture diagram similar to reference image.
        
        Layout:
        - Row 1: Data Ingestion (3 boxes)
        - Row 2: Preprocessing (3 boxes)
        - Row 3: Causal Discovery (center with DAG)
        - Row 4: Causal Inference (3 boxes)
        - Row 5: Counterfactual (4 boxes)
        - Row 6: Alpha Generation (3 boxes)
        - Row 7: Output (2 boxes)
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Create grid
        gs = fig.add_gridspec(8, 6, hspace=0.4, wspace=0.3)
        
        # === ROW 1: DATA INGESTION LAYER ===
        ax_data = fig.add_subplot(gs[0, :])
        ax_data.set_xlim(0, 100)
        ax_data.set_ylim(0, 10)
        ax_data.axis('off')
        
        # Title
        ax_data.text(50, 9, '1. DATA INGESTION LAYER', 
                    fontsize=14, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor=self.colors['data_layer'], 
                             edgecolor=self.colors['border_data'], linewidth=2))
        
        # Three data boxes
        data_boxes = [
            (10, 'Market Data\n• Prices\n• Volumes\n• Order Flow'),
            (40, 'Fundamental\n• Earnings\n• Splits\n• Dividends'),
            (70, 'News & Text\n• Articles\n• Sentiment\n• Social Media')
        ]
        
        for x, text in data_boxes:
            box = FancyBboxPatch((x-8, 1), 16, 6, 
                                boxstyle="round,pad=0.1", 
                                facecolor=self.colors['data_layer'],
                                edgecolor=self.colors['border_data'], linewidth=2)
            ax_data.add_patch(box)
            ax_data.text(x, 4, text, ha='center', va='center', fontsize=9)
        
        # === ROW 2: PREPROCESSING ===
        ax_preprocess = fig.add_subplot(gs[1, :])
        ax_preprocess.set_xlim(0, 100)
        ax_preprocess.set_ylim(0, 10)
        ax_preprocess.axis('off')
        
        ax_preprocess.text(50, 9, '2. PREPROCESSING & FEATURE ENGINEERING', 
                          fontsize=14, fontweight='bold', ha='center',
                          bbox=dict(boxstyle='round', facecolor=self.colors['preprocess'],
                                   edgecolor=self.colors['border_preprocess'], linewidth=2))
        
        prep_boxes = [
            (15, 'Text Processing\nFinBERT Embeddings'),
            (50, 'Time Series\nReturns, Volatility,\nNormalization'),
            (85, 'Feature Alignment\nUnified Matrix\nPrices + Sentiment')
        ]
        
        for x, text in prep_boxes:
            box = FancyBboxPatch((x-12, 1), 24, 6,
                                boxstyle="round,pad=0.1",
                                facecolor=self.colors['preprocess'],
                                edgecolor=self.colors['border_preprocess'], linewidth=2)
            ax_preprocess.add_patch(box)
            ax_preprocess.text(x, 4, text, ha='center', va='center', fontsize=9)
        
        # === ROW 3: TEMPORAL CAUSAL DISCOVERY ===
        ax_discovery = fig.add_subplot(gs[2:4, :])
        ax_discovery.set_xlim(0, 100)
        ax_discovery.set_ylim(0, 20)
        ax_discovery.axis('off')
        
        # Title
        ax_discovery.text(50, 18, '3. TEMPORAL CAUSAL DISCOVERY ENGINE', 
                         fontsize=14, fontweight='bold', ha='center',
                         bbox=dict(boxstyle='round', facecolor=self.colors['discovery'],
                                  edgecolor=self.colors['border_discovery'], linewidth=2))
        
        # Left: PCMCI+ box
        pcmci_box = FancyBboxPatch((5, 2), 20, 14,
                                  boxstyle="round,pad=0.1",
                                  facecolor=self.colors['discovery'],
                                  edgecolor=self.colors['border_discovery'], linewidth=2)
        ax_discovery.add_patch(pcmci_box)
        ax_discovery.text(15, 12, 'PCMCI+', ha='center', fontsize=11, fontweight='bold')
        ax_discovery.text(15, 9, 'Lagged &\nContemporaneous\nCausal Discovery', 
                        ha='center', va='center', fontsize=9)
        
        # Center: Causal Graph DAG
        self._draw_causal_graph_dag(ax_discovery, causal_graph, center=(50, 9), scale=15)
        
        # Right: Structure learning
        structure_box = FancyBboxPatch((75, 2), 20, 14,
                                      boxstyle="round,pad=0.1",
                                      facecolor=self.colors['discovery'],
                                      edgecolor=self.colors['border_discovery'], linewidth=2)
        ax_discovery.add_patch(structure_box)
        ax_discovery.text(85, 12, 'Structure Learning', ha='center', fontsize=11, fontweight='bold')
        ax_discovery.text(85, 9, 'NOTEARS / DAG\nContinuous\nOptimization', 
                        ha='center', va='center', fontsize=9)
        
        # === ROW 4: CAUSAL INFERENCE ===
        ax_inference = fig.add_subplot(gs[4, :])
        ax_inference.set_xlim(0, 100)
        ax_inference.set_ylim(0, 10)
        ax_inference.axis('off')
        
        ax_inference.text(50, 9, '4. CAUSAL INFERENCE LAYER (do-calculus)', 
                         fontsize=14, fontweight='bold', ha='center',
                         bbox=dict(boxstyle='round', facecolor=self.colors['inference'],
                                  edgecolor=self.colors['border_inference'], linewidth=2))
        
        inference_boxes = [
            (20, 'Confounder\nIdentification\nBackdoor Criterion'),
            (50, 'do(X=x)\nIntervention\nPearl\'s do-calculus'),
            (80, 'Causal Effect\nEstimation\nATE / CATE')
        ]
        
        for x, text in inference_boxes:
            box = FancyBboxPatch((x-12, 1), 24, 6,
                                boxstyle="round,pad=0.1",
                                facecolor=self.colors['inference'],
                                edgecolor=self.colors['border_inference'], linewidth=2)
            ax_inference.add_patch(box)
            ax_inference.text(x, 4, text, ha='center', va='center', fontsize=9)
        
        # === ROW 5: COUNTERFACTUAL SIMULATION ===
        ax_counter = fig.add_subplot(gs[5, :])
        ax_counter.set_xlim(0, 100)
        ax_counter.set_ylim(0, 10)
        ax_counter.axis('off')
        
        ax_counter.text(50, 9, '5. COUNTERFACTUAL SIMULATION ENGINE', 
                       fontsize=14, fontweight='bold', ha='center',
                       bbox=dict(boxstyle='round', facecolor=self.colors['counterfactual'],
                                edgecolor=self.colors['border_counter'], linewidth=2))
        
        counter_boxes = [
            (12, 'Historical\nEvent Replay'),
            (35, 'Scenario\nBuilder'),
            (58, 'Counterfactual\nEstimation\nY_{X←x}'),
            (82, 'Validation &\nBacktest')
        ]
        
        for x, text in counter_boxes:
            box = FancyBboxPatch((x-10, 1), 20, 6,
                                boxstyle="round,pad=0.1",
                                facecolor=self.colors['counterfactual'],
                                edgecolor=self.colors['border_counter'], linewidth=2)
            ax_counter.add_patch(box)
            ax_counter.text(x, 4, text, ha='center', va='center', fontsize=9)
        
        # === ROW 6: ALPHA GENERATION ===
        ax_alpha = fig.add_subplot(gs[6, :])
        ax_alpha.set_xlim(0, 100)
        ax_alpha.set_ylim(0, 10)
        ax_alpha.axis('off')
        
        ax_alpha.text(50, 9, '6. ALPHA GENERATION & DECISION LAYER', 
                     fontsize=14, fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round', facecolor=self.colors['alpha'],
                              edgecolor=self.colors['border_alpha'], linewidth=2))
        
        alpha_boxes = [
            (20, 'Causal\nSignals\n• Direction\n• Strength\n• Confidence'),
            (50, 'Trading\nSignals\n• Risk Scores\n• Event Impact\n• Regime Aware'),
            (80, 'Portfolio &\nRisk Management\n• Position Sizing\n• Optimization')
        ]
        
        for x, text in alpha_boxes:
            box = FancyBboxPatch((x-12, 1), 24, 6,
                                boxstyle="round,pad=0.1",
                                facecolor=self.colors['alpha'],
                                edgecolor=self.colors['border_alpha'], linewidth=2)
            ax_alpha.add_patch(box)
            ax_alpha.text(x, 4, text, ha='center', va='center', fontsize=9)
        
        # === ROW 7: OUTPUT & VISUALIZATION ===
        ax_output = fig.add_subplot(gs[7, :])
        ax_output.set_xlim(0, 100)
        ax_output.set_ylim(0, 10)
        ax_output.axis('off')
        
        ax_output.text(50, 9, '7. OUTPUT & VISUALIZATION', 
                      fontsize=14, fontweight='bold', ha='center',
                      bbox=dict(boxstyle='round', facecolor=self.colors['output'],
                               edgecolor=self.colors['border_output'], linewidth=2))
        
        output_boxes = [
            (30, '📊 Causal Graph Explorer\nEvent Impact Dashboard\nAlpha/Signal Dashboard'),
            (70, '📈 Backtest Reports\nAlerts & Notifications\nAPI Layer (REST/GraphQL)')
        ]
        
        for x, text in output_boxes:
            box = FancyBboxPatch((x-18, 1), 36, 6,
                                boxstyle="round,pad=0.1",
                                facecolor=self.colors['output'],
                                edgecolor=self.colors['border_output'], linewidth=2)
            ax_output.add_patch(box)
            ax_output.text(x, 4, text, ha='center', va='center', fontsize=9)
        
        # Add legend for node types
        legend_elements = [
            mpatches.Patch(facecolor=self.colors['node_news'], label='News/Sentiment'),
            mpatches.Patch(facecolor=self.colors['node_macro'], label='Macro Variables'),
            mpatches.Patch(facecolor=self.colors['node_stock'], label='Stocks/Assets'),
            mpatches.Patch(facecolor=self.colors['node_technical'], label='Technical Indicators')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
                  bbox_to_anchor=(0.5, -0.02), fontsize=10)
        
        plt.suptitle('CANDLE/FCIS - FINANCIAL CAUSAL INTELLIGENCE SYSTEM\nFrom Correlation to Causation to Actionable Alpha',
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Architecture diagram saved to {save_path}")
        
        return fig
    
    def _draw_causal_graph_dag(
        self, 
        ax, 
        graph: nx.DiGraph, 
        center: Tuple[float, float] = (50, 9),
        scale: float = 15
    ):
        """Draw the causal DAG in the center of the architecture."""
        cx, cy = center
        
        # Draw background circle for the graph
        circle = Circle(center, scale, facecolor='white', 
                       edgecolor=self.colors['border_discovery'], linewidth=3, alpha=0.9)
        ax.add_patch(circle)
        
        # Title for the graph section
        ax.text(cx, cy + scale - 2, 'Causal Graph (DAG)', 
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=self.colors['discovery'], alpha=0.8))
        
        if len(graph.nodes) == 0:
            ax.text(cx, cy, 'No graph data\nRun causal discovery first', 
                   ha='center', va='center', fontsize=9, style='italic')
            return
        
        # Create layout for nodes
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        
        # Position nodes in a circle
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        
        # Categorize nodes
        sentiment_nodes = [n for n in nodes if 'sentiment' in n.lower()]
        stock_nodes = [n for n in nodes if not any(x in n.lower() for x in ['sentiment', 'news', 'count'])]
        other_nodes = [n for n in nodes if n not in sentiment_nodes and n not in stock_nodes]
        
        # Assign positions by category
        pos = {}
        
        # Sentiment nodes at top
        if sentiment_nodes:
            sentiment_angles = np.linspace(np.pi/3, 2*np.pi/3, len(sentiment_nodes))
            for i, node in enumerate(sentiment_nodes):
                r = scale * 0.6
                pos[node] = (cx + r * np.cos(sentiment_angles[i]), 
                            cy + r * np.sin(sentiment_angles[i]))
        
        # Stock nodes in middle/bottom
        if stock_nodes:
            stock_angles = np.linspace(4*np.pi/3, 5*np.pi/3, len(stock_nodes))
            for i, node in enumerate(stock_nodes):
                r = scale * 0.6
                pos[node] = (cx + r * np.cos(stock_angles[i]), 
                            cy + r * np.sin(stock_angles[i]))
        
        # Others on sides
        if other_nodes:
            other_angles = np.linspace(-np.pi/6, np.pi/6, len(other_nodes))
            for i, node in enumerate(other_nodes):
                r = scale * 0.5
                pos[node] = (cx + r * np.cos(other_angles[i] + np.pi), 
                            cy + r * np.sin(other_angles[i] + np.pi))
        
        # Draw edges
        for edge in graph.edges(data=True):
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                # Edge color based on weight
                weight = edge[2].get('weight', 0.5)
                lag = edge[2].get('lag', 0)
                
                if weight > 0.7:
                    color = self.colors['edge_positive']
                    width = 2
                elif weight < 0.3:
                    color = self.colors['edge_negative']
                    width = 1
                else:
                    color = self.colors['edge_neutral']
                    width = 1
                
                # Arrow
                arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                       arrowstyle='->', mutation_scale=15,
                                       color=color, linewidth=width,
                                       connectionstyle='arc3,rad=0.1')
                ax.add_patch(arrow)
        
        # Draw nodes
        for node, (x, y) in pos.items():
            # Determine color by type
            if 'sentiment' in node.lower():
                color = self.colors['node_news']
            elif node in stock_nodes:
                color = self.colors['node_stock']
            elif 'technical' in node.lower() or 'ma' in node.lower() or 'rsi' in node.lower():
                color = self.colors['node_technical']
            else:
                color = self.colors['node_macro']
            
            # Draw node circle
            circle = Circle((x, y), 1.5, facecolor=color, 
                          edgecolor='black', linewidth=1.5, zorder=5)
            ax.add_patch(circle)
            
            # Label
            label = node[:8] + '...' if len(node) > 10 else node
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=7, fontweight='bold', zorder=6)
        
        # Legend for edges
        ax.text(cx, cy - scale + 3, 'Edge = Strength of Causal Influence', 
               ha='center', fontsize=8, style='italic')


class DAGVisualizer:
    """Create standalone DAG visualizations."""
    
    def __init__(self):
        self.colors = {
            'sentiment': '#FF6B6B',
            'stock': '#4ECDC4',
            'macro': '#45B7D1',
            'technical': '#FFA07A',
            'edge_positive': '#2ECC71',
            'edge_negative': '#E74C3C',
            'edge_weak': '#95A5A6'
        }
    
    def create_publication_dag(
        self,
        graph: nx.DiGraph,
        title: str = "Discovered Causal Graph",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """Create publication-quality DAG visualization."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get node types
        nodes = list(graph.nodes())
        sentiment_nodes = [n for n in nodes if 'sentiment' in n.lower()]
        stock_nodes = [n for n in nodes if not any(x in n.lower() for x in ['sentiment', 'news'])]
        
        # Create hierarchical layout
        pos = self._hierarchical_layout(graph, sentiment_nodes, stock_nodes)
        
        # Draw edges with different styles
        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            weight = edge[2].get('weight', 0.5)
            lag = edge[2].get('lag', 0)
            
            # Style based on properties
            if weight > 0.7:
                style = 'solid'
                color = self.colors['edge_positive']
                width = 3
            elif weight > 0.4:
                style = 'solid'
                color = self.colors['edge_weak']
                width = 1.5
            else:
                style = 'dashed'
                color = self.colors['edge_weak']
                width = 1
            
            # Draw edge
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle='->', color=color, lw=width,
                                      connectionstyle='arc3,rad=0.1'))
            
            # Add lag label if present
            if lag > 0:
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(mid_x, mid_y + 0.05, f't-{lag}', fontsize=7, 
                       ha='center', color=color, style='italic')
        
        # Draw nodes by type
        node_colors = []
        for node in nodes:
            if 'sentiment' in node.lower():
                node_colors.append(self.colors['sentiment'])
            elif 'technical' in node.lower() or any(x in node for x in ['MA', 'RSI', 'VOL']):
                node_colors.append(self.colors['technical'])
            elif node in stock_nodes:
                node_colors.append(self.colors['stock'])
            else:
                node_colors.append(self.colors['macro'])
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=3000, alpha=0.9, ax=ax,
                              edgecolors='black', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=9, font_weight='bold', ax=ax)
        
        # Title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.colors['sentiment'], label='News/Sentiment', edgecolor='black'),
            mpatches.Patch(facecolor=self.colors['stock'], label='Stock Returns', edgecolor='black'),
            mpatches.Patch(facecolor=self.colors['technical'], label='Technical Indicators', edgecolor='black')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Annotations
        ax.text(0.02, 0.02, 'Solid line = Strong causal link\nDashed = Weak link\nt-N = Lag N days',
               transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"DAG saved to {save_path}")
        
        return fig
    
    def _hierarchical_layout(
        self, 
        graph: nx.DiGraph,
        top_nodes: List[str],
        bottom_nodes: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout with top nodes above bottom nodes."""
        pos = {}
        
        # Position top nodes
        if top_nodes:
            x_positions = np.linspace(0.2, 0.8, len(top_nodes))
            for i, node in enumerate(top_nodes):
                pos[node] = (x_positions[i], 0.8)
        
        # Position bottom nodes
        if bottom_nodes:
            x_positions = np.linspace(0.1, 0.9, len(bottom_nodes))
            for i, node in enumerate(bottom_nodes):
                pos[node] = (x_positions[i], 0.2)
        
        # Position any remaining nodes in middle
        remaining = [n for n in graph.nodes() if n not in pos]
        if remaining:
            x_positions = np.linspace(0.15, 0.85, len(remaining))
            for i, node in enumerate(remaining):
                pos[node] = (x_positions[i], 0.5)
        
        return pos
    
    def create_temporal_dag(
        self,
        graph: nx.DiGraph,
        max_lag: int = 5,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create DAG showing temporal/lagged relationships."""
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # Group nodes by time
        nodes_by_time = {}
        for node in graph.nodes():
            # Find max lag for this node as target
            max_lag_for_node = 0
            for edge in graph.in_edges(node, data=True):
                lag = edge[2].get('lag', 0)
                max_lag_for_node = max(max_lag_for_node, lag)
            nodes_by_time[node] = max_lag_for_node
        
        # Position by time
        pos = {}
        time_groups = {}
        for node, lag in nodes_by_time.items():
            if lag not in time_groups:
                time_groups[lag] = []
            time_groups[lag].append(node)
        
        # Create positions
        for lag, nodes in time_groups.items():
            x_positions = np.linspace(0.1, 0.9, len(nodes))
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], 1 - lag / (max_lag + 1))
        
        # Draw
        for edge in graph.edges(data=True):
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                lag = edge[2].get('lag', 0)
                
                color = plt.cm.viridis(lag / max_lag)
                
                ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Nodes
        node_colors = []
        for node in graph.nodes():
            if 'sentiment' in node.lower():
                node_colors.append(self.colors['sentiment'])
            else:
                node_colors.append(self.colors['stock'])
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=2500, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
        
        # Time labels
        for lag in range(max_lag + 1):
            ax.text(-0.05, 1 - lag / (max_lag + 1), f't-{lag}', 
                   ha='right', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('Temporal Causal Graph (Time-Resolved)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_architecture_summary(
    save_path: str = "results/figures/candle_architecture.png"
):
    """Create the full architecture diagram."""
    # Create sample graph for demonstration
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node('sentiment_mean', type='sentiment')
    G.add_node('sentiment_vol', type='sentiment')
    G.add_node('RELIANCE', type='stock')
    G.add_node('TCS', type='stock')
    G.add_node('INFY', type='stock')
    G.add_node('HDFCBANK', type='stock')
    
    # Add edges with properties
    G.add_edge('sentiment_mean', 'RELIANCE', weight=0.8, lag=1)
    G.add_edge('sentiment_mean', 'TCS', weight=0.7, lag=1)
    G.add_edge('sentiment_mean', 'INFY', weight=0.6, lag=2)
    G.add_edge('RELIANCE', 'HDFCBANK', weight=0.5, lag=0)
    G.add_edge('TCS', 'INFY', weight=0.4, lag=1)
    
    # Create visualization
    viz = ArchitectureVisualizer()
    fig = viz.create_framework_architecture(G, save_path=save_path)
    
    return fig


if __name__ == "__main__":
    # Create architecture diagram
    fig = create_architecture_summary()
    plt.show()
