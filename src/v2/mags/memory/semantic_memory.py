"""
Semantic Memory block.
Stores general world knowledge.
"""
import logging
from mags.memory.base_memory import BaseMemoryBlock
from mags.hyperparameters.hyperparameters import Hyperparameters
from mags.query.anchor_search import MaximumAnchorSearch # Default for semantic
from mags.query.traversal import BFSTraversal, TraversalStoppingCondition # Default for semantic

logger = logging.getLogger(__name__)

class SemanticMemory(BaseMemoryBlock):
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(memory_type="semantic", hyperparams=hyperparams)
        logger.info("SemanticMemory initialized.")

    def _initialize_components(self):
        """Initializes default components for Semantic Memory."""
        # Default anchor search: MaximumAnchor based on importance and coverage
        self.anchor_search_strategy = MaximumAnchorSearch(self.hyperparams, self.lkg)

        # Default traversal: BFS, as semantic knowledge might be broad
        stopping_condition = TraversalStoppingCondition(self.hyperparams, component_name="semantic_traversal_stop")
        self.traversal_strategy = BFSTraversal(self.lkg, self.hyperparams, stopping_condition)

        # Register specific hyperparameters for semantic traversal stopping if needed
        self.hyperparams.register_defaults(
            "semantic_traversal_stop", {
                "min_path_importance_product_threshold": 0.05,
                "max_depth": 4,
            }
        )