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
        """Handles the Matplotlib styling and saving."""
        if not self.graph.nodes:
            logger.warning("Graph is empty. Skipping render.")
            return

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(self.graph, k=0.8, seed=42)

        # Dynamic Node Coloring
        node_colors = []
        for n in self.graph.nodes():
            if n.isupper() and len(n) <= 3: # ISO Country Codes
                node_colors.append('#3498db') # Blue
            elif n in self.name_resolver.values():
                node_colors.append('#e74c3c') # Red (Disease)
            else:
                node_colors.append('#2ecc71') # Green (Drug)

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                               node_size=2500, alpha=0.8)
        
        edge_colors = [self.graph[u][v].get('color', 'gray') for u,v in self.graph.edges()]
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, 
                               width=2, arrowsize=20, alpha=0.5)
        
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight="bold")

        plt.title("HERCULE: Global Surveillance & Therapeutic Response", fontsize=15)
        plt.axis('off')
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {output_path}")
        plt.close()