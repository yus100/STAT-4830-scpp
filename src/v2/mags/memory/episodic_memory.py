"""
Episodic Memory block.
Stores specific events and experiences, often temporal.
"""
import logging
from mags.memory.base_memory import BaseMemoryBlock
from mags.hyperparameters.hyperparameters import Hyperparameters
# Choose appropriate defaults, e.g., maybe DFS to follow event sequences
from mags.query.anchor_search import MaximumAnchorSearch # Or one that favors recency/surprise
from mags.query.traversal import DFSTraversal, TraversalStoppingCondition

logger = logging.getLogger(__name__)

class EpisodicMemory(BaseMemoryBlock):
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(memory_type="episodic", hyperparams=hyperparams)
        logger.info("EpisodicMemory initialized.")

    def _initialize_components(self):
        """Initializes default components for Episodic Memory."""
        # Anchor search could prioritize recent or surprising nodes for episodic
        self.anchor_search_strategy = MaximumAnchorSearch(self.hyperparams, self.lkg) # Placeholder

        # Traversal: DFS might be good for following sequences of events
        stopping_condition = TraversalStoppingCondition(self.hyperparams, component_name="episodic_traversal_stop")
        self.traversal_strategy = DFSTraversal(self.lkg, self.hyperparams, stopping_condition)

        self.hyperparams.register_defaults(
            "episodic_traversal_stop", {
                "min_path_importance_product_threshold": 0.02,
                "max_depth": 6, # May go deeper for episodes
            }
        )

        # Episodic memory might have specific modifiers related to time or sequence
        # e.g., a modifier that strengthens sequential links.