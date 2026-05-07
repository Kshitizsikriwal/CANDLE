"""Visualization module for CANDLE framework.

Creates plots and interactive visualizations of:
- Causal graphs
- Time series with interventions
- Counterfactual comparisons
- Causal effect estimates
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class CANDLEVisualizer:
    """Visualization tools for CANDLE results."""
    
    def __init__(self, figures_path: str = "results/figures"):
        self.figures_path = figures_path
    
    def plot_causal_graph(
        self, 
        graph: nx.DiGraph,
        title: str = "Causal Graph",
        figsize: tuple = (14, 10),
        save_path: Optional[str] = None
    ):
        """Plot the discovered causal graph."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Node colors based on type
        node_colors = []
        for node in graph.nodes():
            if 'sentiment' in node.lower():
                node_colors.append('#FF6B6B')  # Red for sentiment
            elif any(x in node.upper() for x in ['RELIANCE', 'TCS', 'INFY', 'HDFC']):
                node_colors.append('#4ECDC4')  # Teal for stocks
            else:
                node_colors.append('#45B7D1')  # Blue for others
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos, 
            node_color=node_colors,
            node_size=2000,
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges with weights
        edges = graph.edges(data=True)
        edge_weights = [d.get('weight', 1) for (_, _, d) in edges]
        
        nx.draw_networkx_edges(
            graph, pos,
            width=[w * 2 for w in edge_weights],
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Labels
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold', ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Sentiment'),
            Patch(facecolor='#4ECDC4', label='Stock Returns'),
            Patch(facecolor='#45B7D1', label='Other')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved causal graph to {save_path}")
        
        return fig
    
    def plot_causal_effects(
        self,
        effects_df: pd.DataFrame,
        title: str = "Causal Effect Estimates",
        figsize: tuple = (12, 8),
        save_path: Optional[str] = None
    ):
        """Plot causal effect estimates as heatmap."""
        # Pivot for heatmap
        pivot = effects_df.pivot(index='treatment', columns='outcome', values='ate')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Average Treatment Effect (ATE)'}
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Outcome Variable', fontsize=12)
        ax.set_ylabel('Treatment Variable', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_time_series_with_interventions(
        self,
        data: pd.DataFrame,
        interventions: List[Dict],
        variables: List[str],
        save_path: Optional[str] = None
    ):
        """Plot time series with intervention markers."""
        fig, axes = plt.subplots(len(variables), 1, figsize=(14, 3 * len(variables)), sharex=True)
        
        if len(variables) == 1:
            axes = [axes]
        
        for idx, var in enumerate(variables):
            ax = axes[idx]
            
            # Plot time series
            ax.plot(data.index, data[var], label=var, alpha=0.7)
            
            # Mark interventions
            for intervention in interventions:
                if var in intervention:
                    date = intervention.get('date')
                    if date and date in data.index:
                        ax.axvline(x=date, color='red', linestyle='--', alpha=0.5)
                        ax.scatter([date], [data.loc[date, var]], color='red', s=100, zorder=5)
            
            ax.set_ylabel(var)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        fig.suptitle('Time Series with Interventions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_counterfactual_comparison(
        self,
        factual: Dict[str, float],
        counterfactual: Dict[str, float],
        intervention: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """Plot factual vs counterfactual comparison."""
        variables = list(factual.keys())
        
        factual_vals = [factual[v] for v in variables]
        counterfactual_vals = [counterfactual[v] for v in variables]
        
        x = np.arange(len(variables))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, factual_vals, width, label='Factual', alpha=0.8)
        bars2 = ax.bar(x + width/2, counterfactual_vals, width, label='Counterfactual', alpha=0.8)
        
        ax.set_xlabel('Variables')
        ax.set_ylabel('Value')
        ax.set_title('Factual vs Counterfactual Outcomes')
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_causal_graph(
        self, 
        graph: nx.DiGraph,
        title: str = "Interactive Causal Graph"
    ) -> go.Figure:
        """Create interactive Plotly causal graph."""
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = edge[2].get('weight', 1)
            lag = edge[2].get('lag', 0)
            edge_text.append(f"Weight: {weight:.3f}<br>Lag: {lag}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            mode='lines',
            text=edge_text
        )
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            node_text.append(f"{node}<br>In: {in_degree}<br>Out: {out_degree}")
            
            # Color based on type
            if 'sentiment' in node.lower():
                node_color.append('#FF6B6B')
            elif any(x in node.upper() for x in ['RELIANCE', 'TCS', 'INFY']):
                node_color.append('#4ECDC4')
            else:
                node_color.append('#45B7D1')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(graph.nodes()),
            textposition="top center",
            marker=dict(
                size=30,
                color=node_color,
                line_width=2
            ),
            hovertext=node_text
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def plot_what_if_analysis(
        self,
        scenarios_df: pd.DataFrame,
        intervention_var: str,
        outcome_var: str,
        save_path: Optional[str] = None
    ):
        """Plot what-if analysis results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Counterfactual vs Intervention value
        ax1.plot(
            scenarios_df['intervention_value'],
            scenarios_df[f'{outcome_var}_counterfactual'],
            marker='o',
            label='Counterfactual'
        )
        ax1.axhline(
            y=scenarios_df[f'{outcome_var}_factual'].iloc[0],
            color='red',
            linestyle='--',
            label='Factual'
        )
        ax1.set_xlabel(f'{intervention_var} (Intervention Value)')
        ax1.set_ylabel(f'{outcome_var} (Predicted)')
        ax1.set_title('Counterfactual Outcomes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Difference from factual
        ax2.bar(
            scenarios_df['intervention_value'],
            scenarios_df[f'{outcome_var}_difference'],
            alpha=0.7
        )
        ax2.axhline(y=0, color='black', linestyle='-')
        ax2.set_xlabel(f'{intervention_var} (Intervention Value)')
        ax2.set_ylabel(f'Difference in {outcome_var}')
        ax2.set_title('Causal Effect (Difference from Factual)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_summary_report(
    pipeline_results: Dict,
    output_path: str = "results/summary_report.html"
):
    """Create HTML summary report of CANDLE results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CANDLE/FCIS Results Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
            .metric-label {{ font-size: 12px; color: #7f8c8d; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #3498db; color: white; }}
            tr:hover {{ background: #f5f5f5; }}
            .path {{ font-family: monospace; background: #ecf0f1; padding: 2px 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CANDLE/FCIS Analysis Report</h1>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Data Points</div>
                    <div class="metric-value">{pipeline_results.get('data_points', 'N/A')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Variables</div>
                    <div class="metric-value">{pipeline_results.get('n_variables', 'N/A')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Causal Edges</div>
                    <div class="metric-value">{pipeline_results.get('n_causal_edges', 'N/A')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Duration</div>
                    <div class="metric-value">{pipeline_results.get('duration_seconds', 'N/A'):.1f}s</div>
                </div>
            </div>
            
            <h2>Output Files</h2>
            <table>
                <tr><th>File Type</th><th>Path</th></tr>
    """
    
    for name, path in pipeline_results.get('output_paths', {}).items():
        html_content += f"<tr><td>{name}</td><td class='path'>{path}</td></tr>"
    
    html_content += """
            </table>
            
            <h2>Methodology</h2>
            <p>This analysis used the CANDLE framework with three pillars:</p>
            <ul>
                <li><strong>Causal Discovery:</strong> PCMCI+ algorithm to identify causal relationships</li>
                <li><strong>Causal Inference:</strong> Pearl's do-calculus for estimating intervention effects</li>
                <li><strong>Counterfactual Simulation:</strong> "What-if" scenario analysis</li>
            </ul>
            
            <p><em>Report generated by CANDLE/FCIS v1.0</em></p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Summary report saved to {output_path}")
