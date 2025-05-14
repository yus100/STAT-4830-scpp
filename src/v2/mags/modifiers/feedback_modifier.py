"""
Feedback Modifier: Adjusts node importance based on external feedback.
"""
import logging
from typing import Dict
from mags.modifiers.base_modifier import BaseModifier, ModifierRunType
from mags.graph.lkg import LiquidKnowledgeGraph
from mags.hyperparameters.hyperparameters import Hyperparameters

logger = logging.getLogger(__name__)

class FeedbackModifier(BaseModifier):
    """
    Adjusts importance scores of specified nodes based on feedback signals
    (e.g., from a model rating usefulness).
    """
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams, run_type=ModifierRunType.AFTER_QUERY) # Or DETACHED
        self.hyperparams.register_defaults(
            self._component_name,
            {
                "positive_feedback_multiplier": 1.2,
                "negative_feedback_multiplier": 0.8,
                "max_importance_score": 100.0,
                "min_importance_score": 0.01
            }
        )

    def apply(self,
              graph: LiquidKnowledgeGraph,
              feedback_scores: Dict[int, float],
              *args, **kwargs) -> None:
        """
        Applies feedback to nodes.

        Args:
            graph: The LiquidKnowledgeGraph instance.
            feedback_scores: A dictionary mapping node_id to a feedback score.
                             Positive scores indicate positive feedback, negative for negative.
                             The magnitude can be used for scaling if desired.
                             For simplicity, here we just use positive/negative.
        """
        pos_multiplier = self._get_param("positive_feedback_multiplier")
        neg_multiplier = self._get_param("negative_feedback_multiplier")
        max_importance = self._get_param("max_importance_score")
        min_importance = self._get_param("min_importance_score")

        nodes_updated = 0
        for node_id, score in feedback_scores.items():
            node = graph.get_node(node_id)
            if node:
                if score > 0: # Positive feedback
                    node.importance_score = min(node.importance_score * pos_multiplier, max_importance)
                elif score < 0: # Negative feedback
                    node.importance_score = max(node.importance_score * neg_multiplier, min_importance)
                # score == 0 could mean no change or neutral feedback
                nodes_updated += 1
                graph.change_count +=1


        logger.info(f"Applied feedback to {nodes_updated} nodes.")
        if not args and not kwargs:
            pass