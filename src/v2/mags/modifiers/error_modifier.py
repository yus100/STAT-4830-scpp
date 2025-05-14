"""
Error Calculator Modifier: Calculates a heuristic error/complexity score for the graph.
"""
import logging
from mags.modifiers.base_modifier import BaseModifier, ModifierRunType
from mags.graph.lkg import LiquidKnowledgeGraph
from mags.hyperparameters.hyperparameters import Hyperparameters

logger = logging.getLogger(__name__)

class ErrorCalculatorModifier(BaseModifier):
    """
    Calculates a heuristic "error" or complexity score for the graph.
    This could be a function of the number of changes (churn) and the size of the graph.
    The definition of "error" is application-specific.
    """
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams, run_type=ModifierRunType.DETACHED) # Typically for analysis
        self.hyperparams.register_defaults(
            self._component_name, # "errorcalculator"
            {
                "change_weight": 0.1, # How much each recorded change contributes to error
                "node_size_weight": 0.01, # How much each node contributes
                "edge_size_weight": 0.005  # How much each edge contributes
            }
        )

    def apply(self, graph: LiquidKnowledgeGraph, *args, **kwargs) -> float:
        """
        Calculates and returns the error score.
        This modifier doesn't change the graph, it just computes a value.
        The `apply` method for analysis modifiers might return the computed value.
        """
        change_w = self._get_param("change_weight")
        node_w = self._get_param("node_size_weight")
        edge_w = self._get_param("edge_size_weight")

        # graph.change_count should be a counter incremented by LKG operations
        # and other modifiers when they alter the graph.
        error_score = (graph.change_count * change_w +
                       graph.get_node_count() * node_w +
                       graph.get_edge_count() * edge_w)

        logger.info(f"Calculated graph error/complexity score: {error_score:.4f} "
                    f"(changes: {graph.change_count}, nodes: {graph.get_node_count()}, edges: {graph.get_edge_count()})")
        
        if not args and not kwargs: # Suppress unused variable warning if no args are expected
            pass
        return error_score # Return the calculated score