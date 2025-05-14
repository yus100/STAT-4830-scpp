"""
Pruning Modifiers: Remove weak nodes/edges or prune to meet size constraints.
"""
import logging
from typing import List
from mags.modifiers.base_modifier import BaseModifier, ModifierRunType
from mags.graph.lkg import LiquidKnowledgeGraph
from mags.graph.node import Node
from mags.graph.edge import Edge
from mags.hyperparameters.hyperparameters import Hyperparameters

logger = logging.getLogger(__name__)

class Prune1Modifier(BaseModifier):
    """
    Prune1: Removes edges with strength below a threshold and
            nodes (non-permanent) with importance below a different threshold.
    """
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams, run_type=ModifierRunType.DETACHED)
        self.hyperparams.register_defaults(
            self._component_name, # Will be "prune1"
            {
                "edge_strength_prune_threshold": 0.05,
                "node_importance_prune_threshold": 0.1
            }
        )

    def apply(self, graph: LiquidKnowledgeGraph, *args, **kwargs) -> None:
        edge_threshold = self._get_param("edge_strength_prune_threshold")
        node_threshold = self._get_param("node_importance_prune_threshold")

        # Prune edges first
        edges_to_prune_ids = [
            edge.id for edge in graph.get_all_edges() if edge.strength_score < edge_threshold
        ]
        for edge_id in edges_to_prune_ids:
            graph.remove_edge(edge_id)
        logger.info(f"Prune1: Removed {len(edges_to_prune_ids)} edges below strength {edge_threshold}.")

        # Prune nodes
        # Note: Removing nodes might make some edges dangling if not handled by graph.remove_node
        # graph.remove_node should handle removing incident edges.
        nodes_to_prune_ids = [
            node.id for node in graph.get_all_nodes()
            if not node.is_permanent and node.importance_score < node_threshold
        ]
        for node_id in nodes_to_prune_ids:
            graph.remove_node(node_id) # remove_node also increments graph.change_count
        logger.info(f"Prune1: Removed {len(nodes_to_prune_ids)} non-permanent nodes below importance {node_threshold}.")
        if not args and not kwargs:
            pass


class Prune2Modifier(BaseModifier):
    """
    Prune2: If the number of nodes/edges exceeds a threshold,
            prunes the lowest importance/strength ones (non-permanent for nodes)
            until the maximum is reached.
    """
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams, run_type=ModifierRunType.DETACHED)
        # These max values can also be taken from lkg's own hyperparameters
        self.hyperparams.register_defaults(
            self._component_name, # "prune2"
            {
                "max_nodes_threshold": None, # If None, uses lkg.max_nodes
                "max_edges_threshold": None  # If None, uses lkg.max_edges
            }
        )

    def apply(self, graph: LiquidKnowledgeGraph, *args, **kwargs) -> None:
        max_nodes = self._get_param("max_nodes_threshold") or \
                    self.hyperparams.get_component_param("lkg", "max_nodes", 10000)
        max_edges = self._get_param("max_edges_threshold") or \
                    self.hyperparams.get_component_param("lkg", "max_edges", 50000)

        # Prune nodes if over limit
        num_nodes_to_prune = graph.get_node_count() - max_nodes
        if num_nodes_to_prune > 0:
            # Sort non-permanent nodes by importance (ascending)
            eligible_nodes: List[Node] = sorted(
                [node for node in graph.get_all_nodes() if not node.is_permanent],
                key=lambda n: n.importance_score
            )
            nodes_pruned_count = 0
            for i in range(min(num_nodes_to_prune, len(eligible_nodes))):
                graph.remove_node(eligible_nodes[i].id)
                nodes_pruned_count += 1
            logger.info(f"Prune2: Pruned {nodes_pruned_count} nodes to meet max_nodes limit of {max_nodes}.")

        # Prune edges if over limit
        num_edges_to_prune = graph.get_edge_count() - max_edges
        if num_edges_to_prune > 0:
            # Sort edges by strength (ascending)
            eligible_edges: List[Edge] = sorted(
                graph.get_all_edges(),
                key=lambda e: e.strength_score
            )
            edges_pruned_count = 0
            for i in range(min(num_edges_to_prune, len(eligible_edges))):
                graph.remove_edge(eligible_edges[i].id)
                edges_pruned_count += 1
            logger.info(f"Prune2: Pruned {edges_pruned_count} edges to meet max_edges limit of {max_edges}.")
        if not args and not kwargs:
            pass