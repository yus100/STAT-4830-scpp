import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import Dict, Any

class GraphVisualizer:
    """
    A modular class to visualize graphs with interactive node details and edge weights.
    """

    def __init__(self, graph: nx.Graph, node_details: Dict[Any, Dict[str, Any]] = None):
        """
        Initializes the GraphVisualizer with a graph and optional node details.

        Args:
            graph (nx.Graph): The graph to visualize (networkx Graph object).
            node_details (Dict[Any, Dict[str, Any]], optional): A dictionary mapping node identifiers to their details.
                                                                 Defaults to None.
        """
        self.graph = graph
        self.node_details = node_details if node_details else {}
        self.popup_annotation = None
        self.edge_labels = None
        self.node_positions = None
        self.fig = None
        self.ax = None

    def _create_popup(self, node):
        """
        Creates the popup annotation for a node.

        Args:
            node: The node identifier.

        Returns:
            matplotlib.text.Annotation: The popup annotation object.
        """
        details_text = f"Node: {node}\n"
        if node in self.node_details:
            for key, value in self.node_details[node].items():
                details_text += f"{key}: {value}\n"
        else:
            details_text += "No details available."

        x, y = self.node_positions[node]
        return self.ax.annotate(
            details_text,
            xy=(x, y),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
        )

    def _on_node_hover(self, event):
        """
        Handles node hover events to display node details in a popup.

        Args:
            event: The hover event.
        """
        if event.inaxes == self.ax:
            cont, ind = event.contains(self.nodes)
            if cont:
                node_index = ind["ind"][0]
                hovered_node = list(self.graph.nodes)[node_index]

                if self.popup_annotation:
                    self.popup_annotation.remove()
                    self.popup_annotation = None

                self.popup_annotation = self._create_popup(hovered_node)
                self.fig.canvas.draw_idle()
            else:
                if self.popup_annotation:
                    self.popup_annotation.remove()
                    self.popup_annotation = None
                    self.fig.canvas.draw_idle()

    def visualize(self, layout='spring_layout', node_color='skyblue', node_size=500,
                  edge_color='gray', edge_width=1.0, font_size=8, font_color='black'):
        """
        Visualizes the graph with customizable layout and styles.

        Args:
            layout (str, optional): Layout algorithm for node positioning ('spring_layout', ' Kamada-Kawai', 'circular_layout', etc.). Defaults to 'spring_layout'.
            node_color (str, optional): Color of the nodes. Defaults to 'skyblue'.
            node_size (int, optional): Size of the nodes. Defaults to 500.
            edge_color (str, optional): Color of the edges. Defaults to 'gray'.
            edge_width (float, optional): Width of the edges. Defaults to 1.0.
            font_size (int, optional): Font size for edge weights. Defaults to 8.
            font_color (str, optional): Font color for edge weights. Defaults to 'black'.
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Node Layout
        if layout == 'spring_layout':
            self.node_positions = nx.spring_layout(self.graph)
        elif layout == 'kamada_kawai_layout':
            self.node_positions = nx.kamada_kawai_layout(self.graph)
        elif layout == 'circular_layout':
            self.node_positions = nx.circular_layout(self.graph)
        else:
            self.node_positions = nx.spring_layout(self.graph)  # Default layout

        # Draw Nodes
        self.nodes = nx.draw_networkx_nodes(self.graph, self.node_positions, node_color=node_color, node_size=node_size, ax=self.ax)

        # Draw Edges
        nx.draw_networkx_edges(self.graph, self.node_positions, edge_color=edge_color, width=edge_width, ax=self.ax)

        # Edge Labels (Weights)
        self.edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, self.node_positions, edge_labels=self.edge_labels, font_size=font_size, font_color=font_color, ax=self.ax)

        # Node Labels (optional - can be customized)
        nx.draw_networkx_labels(self.graph, self.node_positions, ax=self.ax)

        self.ax.set_title("Graph Visualization")
        self.ax.axis('off')  # Turn off axis numbers and ticks

        # Connect hover event for node popups
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_node_hover)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example Usage:
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(['A', 'B', 'C', 'D'])
    G.add_edges_from([('A', 'B', {'weight': 4}), ('B', 'C', {'weight': 2}), ('C', 'D', {'weight': 1}), ('D', 'A', {'weight': 3})])

    # Sample node details
    node_details_info = {
        'A': {'type': 'Start Node', 'value': 10},
        'B': {'type': 'Intermediate', 'value': 25},
        'C': {'type': 'Intermediate', 'value': 15},
        'D': {'type': 'End Node', 'value': 5}
    }

    # Initialize and visualize the graph
    visualizer = GraphVisualizer(G, node_details=node_details_info)
    visualizer.visualize()
