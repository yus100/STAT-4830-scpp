"""
Hebbian Modifier: Strengthens edges between co-activated nodes.
"""
import logging
from typing import Set, Tuple
from mags.modifiers.base_modifier import BaseModifier, ModifierRunType
from mags.graph.lkg import LiquidKnowledgeGraph
from mags.hyperparameters.hyperparameters import Hyperparameters

logger = logging.getLogger(__name__)

class HebbianModifier(BaseModifier):
    """
    Strengthens edges between nodes that were BOTH recalled/activated together.
    "Neurons that fire together, wire together."
    """
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams, run_type=ModifierRunType.AFTER_QUERY)
        self.hyperparams.register_defaults(
            self._component_name,
            {
                "strength_increment": 0.1, # Additive or multiplicative
                "max_strength_score": 1.0
            }
        )

    def apply(self,
              graph: LiquidKnowledgeGraph,
              activated_node_pairs: Set[Tuple[int, int]], # Pairs (node_id1, node_id2) that were co-activated
              *args, **kwargs) -> None:
        """
        Applies Hebbian learning to edges between co-activated nodes.

        Args:
            graph: The LiquidKnowledgeGraph instance.
            activated_node_pairs: A set of tuples, where each tuple (u, v)
                                  represents that node u and node v were
                                  activated together. The modifier will look for
                                  edges u -> v or v -> u.
        """
        increment = self._get_param("strength_increment")
        max_strength = self._get_param("max_strength_score")

        edges_strengthened = 0
        for u_id, v_id in activated_node_pairs:
            # Check for edge u -> v
            edge_uv = graph.get_edge_between_nodes(u_id, v_id)
            if edge_uv:
                edge_uv.strength_score = min(edge_uv.strength_score + increment, max_strength)
                edges_strengthened += 1
                graph.change_count +=1


            # Optionally, check for edge v -> u if the interaction is symmetric
            # Or if the definition of "co-activated" means they influence each other regardless of direction
            edge_vu = graph.get_edge_between_nodes(v_id, u_id)
            if edge_vu and edge_vu != edge_uv: # ensure not double counting if u->v and v->u exist as separate edges
                edge_vu.strength_score = min(edge_vu.strength_score + increment, max_strength)
                edges_strengthened += 1
                graph.change_count +=1

        logger.info(f"Applied Hebbian strengthening to {edges_strengthened} edge instances.")
        if not args and not kwargs:
            pass