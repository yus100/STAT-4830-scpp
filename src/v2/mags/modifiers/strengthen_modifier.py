"""
Strengthen Modifier: Strengthens importance and strength of used items.
"""
import logging
from typing import List, Set
from mags.modifiers.base_modifier import BaseModifier, ModifierRunType
from mags.graph.lkg import LiquidKnowledgeGraph
from mags.hyperparameters.hyperparameters import Hyperparameters

logger = logging.getLogger(__name__)

class StrengthenModifier(BaseModifier):
    """
    Strengthens the importance scores of recalled/used nodes and
    strength scores of edges involved with these nodes.
    """
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams, run_type=ModifierRunType.AFTER_QUERY)
        self.hyperparams.register_defaults(
            self._component_name,
            {
                "node_importance_boost_factor": 1.1, # Multiplicative boost
                "edge_strength_boost_factor": 1.05,  # Multiplicative boost
                "max_importance_score": 100.0,
                "max_strength_score": 1.0
            }
        )

    def apply(self,
              graph: LiquidKnowledgeGraph,
              recalled_node_ids: Set[int],
              recalled_edge_ids: Set[int],
              *args, **kwargs) -> None:
        """
        Applies strengthening to the specified nodes and edges.

        Args:
            graph: The LiquidKnowledgeGraph instance.
            recalled_node_ids: A set of IDs of nodes that were recalled/used.
            recalled_edge_ids: A set of IDs of edges that were part of recalled triplets.
        """
        node_boost = self._get_param("node_importance_boost_factor")
        edge_boost = self._get_param("edge_strength_boost_factor")
        max_importance = self._get_param("max_importance_score")
        max_strength = self._get_param("max_strength_score")

        nodes_strengthened = 0
        for node_id in recalled_node_ids:
            node = graph.get_node(node_id)
            if node:
                node.importance_score = min(node.importance_score * node_boost, max_importance)
                nodes_strengthened += 1
                graph.change_count +=1

        edges_strengthened = 0
        for edge_id in recalled_edge_ids:
            edge = graph.get_edge(edge_id)
            if edge:
                edge.strength_score = min(edge.strength_score * edge_boost, max_strength)
                edges_strengthened += 1
                graph.change_count +=1

        logger.info(f"Applied strengthening: {nodes_strengthened} nodes, {edges_strengthened} edges.")
        if not args and not kwargs:
            pass