import logging
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class HerculeDashboard:
    def __init__(self, disease_map, name_resolver, drug_map=None):
        self.disease_map = disease_map  # Map of WHO Code -> DOID
        self.name_resolver = name_resolver  # Map of WHO Code -> Human Name
        self.drug_map = drug_map or {}  # Map of DB_ID -> Drug Name
        self.graph = nx.DiGraph()

    def _resolve_label(self, label):
        """Standardized label resolver for nodes."""
        # Clean prefix
        code = label.split('::')[-1] if '::' in label else label
        
        # Check if it's a disease (WHO/DOID)
        for ind, doid in self.disease_map.items():
            if doid == label or ind == label:
                return self.name_resolver.get(ind, label)
        
        # Check if it's a drug (DrugBank)
        return self.drug_map.get(code, code)

    def build_surveillance_graph(self, surveillance_triples, all_triples, top_n=3):
        """Constructs the network based on highest-impact countries."""
        logger.info(f"Building dashboard graph for top {top_n} countries")
        
        # Identify top countries by outbreak count
        country_counts = {}
        for h, r, t in surveillance_triples:
            country_counts[h] = country_counts.get(h, 0) + 1
        
        top_countries = [c[0] for c in sorted(country_counts.items(), 
                         key=lambda x: x[1], reverse=True)[:top_n]]
        
        # 1. Add Country -> Disease edges
        disease_nodes = set()
        for h, r, t in all_triples:
            if h in top_countries and r == 'has_active_outbreak':
                self.graph.add_edge(h.split('::')[-1], self._resolve_label(t), 
                                    color='#3498db', type='country_disease')
                disease_nodes.add(t)

        # 2. Add Disease -> Treatment edges
        for h, r, t in all_triples:
            if t in disease_nodes and r in ['CtD', 'CpD']:
                m_name = self._resolve_label(h)
                if len(m_name) < 20:  # Safety filter for long chemical names
                    self.graph.add_edge(self._resolve_label(t), m_name, 
                                        color='#2ecc71', type='disease_drug')

    def render(self, output_path="output/dashboard.png"):
            """Handles the Matplotlib styling with optimized spacing and anti-overlap."""
            if not self.graph.nodes:
                logger.warning("Graph is empty. Skipping render.")
                return

            # 1. Significantly increase figure size for better resolution and spacing
            plt.figure(figsize=(20, 14))
            
            # 2. Optimize Spring Layout: 
            # Increase 'k' to push nodes further apart (default is 1/sqrt(n))
            # Increase 'iterations' for a more stable equilibrium
            pos = nx.spring_layout(self.graph, k=1.5, iterations=100, seed=42)

            # Dynamic Node Coloring (kept from your original logic)
            node_colors = []
            for n in self.graph.nodes():
                if n.isupper() and len(n) <= 3: 
                    node_colors.append('#3498db') # Blue (Country)
                elif n in self.name_resolver.values():
                    node_colors.append('#e74c3c') # Red (Disease)
                else:
                    node_colors.append('#2ecc71') # Green (Drug)

            # 3. Use smaller node sizes and transparent alphas to reduce visual weight
            nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                                node_size=1800, alpha=0.9)
            
            edge_colors = [self.graph[u][v].get('color', 'gray') for u,v in self.graph.edges()]
            nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, 
                                width=1.5, arrowsize=15, alpha=0.4, connectionstyle='arc3,rad=0.1')
            
            # 4. Improve label readability:
            # Use a smaller font and 'clip_on=False' to ensure labels aren't cut off
            nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight="bold", 
                                    font_family="sans-serif")

            plt.title("HERCULE: Global Surveillance & Therapeutic Response", fontsize=20, pad=20)
            plt.axis('off')
            
            # 5. Use 'bbox_inches=tight' to ensure no labels are cut at the edges
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
            logger.info(f"Dashboard saved to {output_path}")
            plt.close()
