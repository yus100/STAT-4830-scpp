"""
Anchor Search algorithms for selecting initial nodes in LKG queries.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Dict, Any, Optional
from mags.graph.lkg import LiquidKnowledgeGraph, Node
from mags.hyperparameters.hyperparameters import Hyperparameters

logger = logging.getLogger(__name__)

class AnchorSearch(ABC):
    """
    Abstract base class for anchor search strategies.
    """
    def __init__(self, hyperparams: Hyperparameters, graph: LiquidKnowledgeGraph):
        self.hyperparams = hyperparams
        self.graph = graph
        self._component_name = self.__class__.__name__.lower().replace("search", "")
        self.hyperparams.register_defaults(
            self._component_name,
            {
                "num_anchors_to_select_N": 3,
            }
        )

    def _get_param(self, key: str, default: Any = None) -> Any:
        return self.hyperparams.get_component_param(self._component_name, key, default)

    @abstractmethod
    def find_anchors(self, query_embedding: Optional[List[float]] = None, **kwargs) -> List[Node]:
        """
        Selects N initial anchor nodes from the graph based on the strategy.

        Args:
            query_embedding: Optional embedding of the query for similarity-based selection.
            **kwargs: Additional strategy-specific parameters.

        Returns:
            A list of N selected anchor Nodes.
        """
        pass


class MaximumAnchorSearch(AnchorSearch):
    """
    MaximumAnchorSearch: Chooses N anchors that have the highest sum of importance scores
    while maximizing the 'area covered'.
    Area is the sum of unique nodes captured within X edges of an anchor node.
    """
    def __init__(self, hyperparams: Hyperparameters, graph: LiquidKnowledgeGraph):
        super().__init__(hyperparams, graph)
        self.hyperparams.register_defaults(
            self._component_name, # "maximumanchor"
            {
                "area_coverage_depth_X": 2, # X edges for area calculation
            }
        )

    def _get_nodes_within_distance(self, start_node_id: int, max_depth: int) -> Set[int]:
        """Helper to find all unique node IDs within max_depth using BFS."""
        if start_node_id not in self.graph.nodes:
            return set()

        q: List[Tuple[int, int]] = [(start_node_id, 0)] # (node_id, depth)
        visited: Set[int] = {start_node_id}
        area_nodes: Set[int] = {start_node_id}

        head = 0
        while head < len(q):
            curr_node_id, depth = q[head]
            head += 1

            if depth >= max_depth:
                continue

            # Consider both outgoing and incoming edges for "area coverage"
            neighbor_edge_pairs = self.graph.adj.get(curr_node_id, []) + \
                                  [(n_id, e_id) for n_id, e_id in self.graph.rev_adj.get(curr_node_id, [])
                                   if self.graph.get_edge(e_id)] # Ensure edge exists

            for neighbor_id, _ in neighbor_edge_pairs:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    area_nodes.add(neighbor_id)
                    q.append((neighbor_id, depth + 1))
        return area_nodes

    def find_anchors(self, query_embedding: Optional[List[float]] = None, **kwargs) -> List[Node]:
        """
        Finds N anchors by iteratively selecting the node that adds the most
        to the (sum of importance scores * new area covered).
        This is a greedy approach.
        """
        num_anchors_N = self._get_param("num_anchors_to_select_N", 3)
        coverage_depth_X = self._get_param("area_coverage_depth_X", 2)

        candidate_nodes: List[Node] = [
            node for node in self.graph.get_all_nodes() if not node.is_permanent # Or consider all
        ]
        # Sort candidates by importance initially as a heuristic or if no query_embedding
        candidate_nodes.sort(key=lambda n: n.importance_score, reverse=True)

        # If query_embedding is provided, could re-rank candidates by similarity * importance
        # For now, this implementation focuses on importance and coverage.

        selected_anchors: List[Node] = []
        covered_area_nodes: Set[int] = set()

        for _ in range(num_anchors_N):
            best_candidate: Optional[Node] = None
            max_score_gain = -float('inf')

            if not candidate_nodes:
                break

            for candidate_node in candidate_nodes:
                if candidate_node.id in [a.id for a in selected_anchors]: # Already selected
                    continue

                # Calculate new area this candidate would cover
                candidate_area = self._get_nodes_within_distance(candidate_node.id, coverage_depth_X)
                newly_covered_area = candidate_area - covered_area_nodes
                
                if not newly_covered_area: # No new coverage
                    score_gain = candidate_node.importance_score * 0.1 # Small bonus for importance itself
                else:
                    # Score: Importance * Number of new nodes covered
                    # More sophisticated: sum of importance of newly covered nodes
                    score_gain = candidate_node.importance_score * len(newly_covered_area)


                # Alternative scoring: sum of importance scores of all nodes in `newly_covered_area`
                # plus candidate_node.importance_score.
                # sum_importance_new_area = sum(self.graph.get_node(nid).importance_score for nid in newly_covered_area if self.graph.get_node(nid))
                # score_gain = candidate_node.importance_score + sum_importance_new_area


                if score_gain > max_score_gain:
                    max_score_gain = score_gain
                    best_candidate = candidate_node
            
            if best_candidate:
                selected_anchors.append(best_candidate)
                best_candidate_area = self._get_nodes_within_distance(best_candidate.id, coverage_depth_X)
                covered_area_nodes.update(best_candidate_area)
                # Remove selected from candidates to avoid re-processing, though the check handles it
                candidate_nodes = [cn for cn in candidate_nodes if cn.id != best_candidate.id]
            else:
                break # No more suitable candidates

        logger.info(f"MaximumAnchorSearch selected {len(selected_anchors)} anchors: {[a.id for a in selected_anchors]}")
        return selected_anchors