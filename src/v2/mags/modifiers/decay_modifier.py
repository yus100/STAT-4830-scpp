"""
Decay Modifier: Decays importance and strength scores in the graph.
"""
import logging
from mags.modifiers.base_modifier import BaseModifier, ModifierRunType
from mags.graph.lkg import LiquidKnowledgeGraph
from mags.hyperparameters.hyperparameters import Hyperparameters

logger = logging.getLogger(__name__)

class DecayModifier(BaseModifier):
    """
    Decays importance scores of non-permanent nodes and strength scores of all edges.
    """
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams, run_type=ModifierRunType.AFTER_QUERY) # Or DETACHED
        self.hyperparams.register_defaults(
            self._component_name,
            {
                "node_importance_decay_factor": 0.95, # Multiplicative decay
                "edge_strength_decay_factor": 0.98,   # Multiplicative decay
                "min_importance_threshold": 0.01, # Importance below this might be pruned later
                "min_strength_threshold": 0.01    # Strength below this might be pruned later
            }
        )

    def apply(self, graph: LiquidKnowledgeGraph, *args, **kwargs) -> None:
        """
        Applies decay to all relevant nodes and edges.
        No specific arguments needed beyond the graph itself.
        """
        node_decay_factor = self._get_param("node_importance_decay_factor", 0.95)
        edge_decay_factor = self._get_param("edge_strength_decay_factor", 0.98)
        min_importance = self._get_param("min_importance_threshold", 0.01)
        min_strength = self._get_param("min_strength_threshold", 0.01)

        nodes_decayed = 0
        for node in graph.get_all_nodes():
            if not node.is_permanent:
                node.importance_score *= node_decay_factor
                if node.importance_score < min_importance:
                    node.importance_score = min_importance # Floor or allow to go to zero for pruning
                nodes_decayed += 1
                graph.change_count +=1


        edges_decayed = 0
        for edge in graph.get_all_edges():
            edge.strength_score *= edge_decay_factor
            if edge.strength_score < min_strength:
                edge.strength_score = min_strength # Floor or allow to go to zero
            edges_decayed += 1
            graph.change_count +=1


        logger.info(f"Applied decay: {nodes_decayed} nodes affected, {edges_decayed} edges affected.")
        if not args and not kwargs: # Suppress unused variable warning if no args are expected
            pass