"""
General Memory block.
A default or combined memory type.
"""
import logging
from mags.memory.base_memory import BaseMemoryBlock
from mags.hyperparameters.hyperparameters import Hyperparameters
from mags.query.anchor_search import MaximumAnchorSearch
from mags.query.traversal import BFSTraversal, TraversalStoppingCondition # Or a hybrid

logger = logging.getLogger(__name__)

class GeneralMemory(BaseMemoryBlock):
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(memory_type="general", hyperparams=hyperparams)
        logger.info("GeneralMemory initialized.")

    def _initialize_components(self):
        """Initializes default components for General Memory."""
        self.anchor_search_strategy = MaximumAnchorSearch(self.hyperparams, self.lkg)

        stopping_condition = TraversalStoppingCondition(self.hyperparams, component_name="general_traversal_stop")
        self.traversal_strategy = BFSTraversal(self.lkg, self.hyperparams, stopping_condition) # Default to BFS

        self.hyperparams.register_defaults(
            "general_traversal_stop", {
                "min_path_importance_product_threshold": 0.03,
                "max_depth": 5,
            }
        )