"""
Permanence Modifier: Makes highly important nodes permanent.
"""
import logging
from mags.modifiers.base_modifier import BaseModifier, ModifierRunType
from mags.graph.lkg import LiquidKnowledgeGraph
from mags.hyperparameters.hyperparameters import Hyperparameters

logger = logging.getLogger(__name__)

class PermanenceModifier(BaseModifier):
    """
    Marks nodes with importance above a certain threshold as permanent,
    preventing their scores from decaying or being easily pruned.
    """
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams, run_type=ModifierRunType.DETACHED) # Can be run periodically
        self.hyperparams.register_defaults(
            self._component_name,
            {
                "permanence_importance_threshold": 50
            }
        )

    def apply(self, graph: LiquidKnowledgeGraph, *args, **kwargs) -> None:
        """
        Checks all non-permanent nodes and makes them permanent if they exceed the threshold.
        """
        threshold = self._get_param("permanence_importance_threshold")
        nodes_made_permanent = 0

        for node in graph.get_all_nodes():
            if not node.is_permanent and node.importance_score >= threshold:
                node.set_permanent(True)
                nodes_made_permanent += 1
                graph.change_count +=1 # Setting permanent is a change

        logger.info(f"{nodes_made_permanent} nodes marked as permanent.")
        if not args and not kwargs:
            pass