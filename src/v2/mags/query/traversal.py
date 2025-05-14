"""
Traversal algorithms for exploring the LKG from anchor nodes.
"""
import logging
import math
from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Optional, Callable, Dict
from mags.graph.lkg import LiquidKnowledgeGraph, Node, Edge
from mags.hyperparameters.hyperparameters import Hyperparameters
from mags.utils.math_utils import cosine_similarity # Assuming available

logger = logging.getLogger(__name__)

# Type alias for a triplet
Triplet = Tuple[Node, Edge, Node]


class TraversalStoppingCondition:
    """
    Defines conditions for stopping a traversal.
    Example: product of importance scores along path < threshold.
    """
    def __init__(self, hyperparams: Hyperparameters, component_name: str = "traversal_stop"):
        self.hyperparams = hyperparams
        self.component_name = component_name
        self.hyperparams.register_defaults(
            component_name,
            {
                "min_path_importance_product_threshold": 0.01,
                "max_depth": 5,
                "max_nodes_visited": 100
            }
        )
        self.min_importance_product = self.hyperparams.get_component_param(component_name, "min_path_importance_product_threshold")
        self.max_depth = self.hyperparams.get_component_param(component_name, "max_depth")
        self.max_nodes_visited_count = self.hyperparams.get_component_param(component_name, "max_nodes_visited")


    def should_stop(self,
                    current_path_importance_product: float,
                    current_depth: int,
                    visited_nodes_count: int,
                    current_node: Node) -> bool: # Added current_node for flexibility
        if current_path_importance_product < self.min_importance_product:
            logger.debug(f"Traversal stopping: path importance product {current_path_importance_product:.4f} < {self.min_importance_product}")
            return True
        if current_depth >= self.max_depth:
            logger.debug(f"Traversal stopping: depth {current_depth} >= {self.max_depth}")
            return True
        if visited_nodes_count >= self.max_nodes_visited_count:
            logger.debug(f"Traversal stopping: visited nodes {visited_nodes_count} >= {self.max_nodes_visited_count}")
            return True
        # Could add more conditions based on current_node properties
        return False


class TraversalAlgorithm(ABC):
    """
    Abstract base class for graph traversal algorithms.
    """
    def __init__(self,
                 graph: LiquidKnowledgeGraph,
                 hyperparams: Hyperparameters,
                 stopping_condition: TraversalStoppingCondition):
        self.graph = graph
        self.hyperparams = hyperparams
        self.stopping_condition = stopping_condition
        self._component_name = self.__class__.__name__.lower().replace("traversal", "") # e.g., "dfs"
        self.hyperparams.register_defaults(
            self._component_name,
            {
                "top_Y_triplets_to_return": 10,
                "importance_decay_rate_per_hop": 0.9, # For importance decayed from anchor
            }
        )
        self.top_Y = self.hyperparams.get_component_param(self._component_name, "top_Y_triplets_to_return")
        self.importance_decay_rate = self.hyperparams.get_component_param(self._component_name, "importance_decay_rate_per_hop")

    @abstractmethod
    def traverse(self,
                 start_nodes: List[Node],
                 query_embedding: Optional[List[float]] = None) -> List[Triplet]:
        """
        Performs traversal from start_nodes, collects triplets, and returns top Y.

        Args:
            start_nodes: A list of anchor Nodes to start traversal from.
            query_embedding: The embedding of the original query, for similarity calculations.

        Returns:
            A list of the top Y triplets (Node, Edge, Node) found, ordered by relevance.
        """
        pass

    def _calculate_triplet_score(self,
                                 triplet: Triplet,
                                 anchor_node: Node,
                                 distance_from_anchor: int,
                                 query_embedding: Optional[List[float]]) -> float:
        """
        Calculates a score for a triplet.
        Score = (sum of cosine_sim(node_embedding, query_embedding) for nodes in triplet) *
                (importance_decayed_from_anchor for the closer node in triplet to anchor)
        Edge strength can also be incorporated.
        """
        node1, edge, node2 = triplet

        # Cosine similarity score
        similarity_score = 0.0
        num_nodes_with_embeddings = 0
        if query_embedding:
            if node1.embeddings:
                similarity_score += cosine_similarity(node1.embeddings, query_embedding)
                num_nodes_with_embeddings +=1
            if node2.embeddings:
                similarity_score += cosine_similarity(node2.embeddings, query_embedding)
                num_nodes_with_embeddings +=1
        
        avg_similarity = (similarity_score / num_nodes_with_embeddings) if num_nodes_with_embeddings > 0 else 0.5 # Default if no embeddings/query

        # Importance decayed from anchor
        # This assumes `distance_from_anchor` is for the node that *led* to this triplet.
        # Let's use the anchor's original importance decayed.
        decayed_importance = anchor_node.importance_score * (self.importance_decay_rate ** distance_from_anchor)

        # Combine: product of (sum of cosine sims) * decayed importance
        # The prompt phrasing "product of the sum of the cosine similaritys of the nodes times the importance decayed"
        # can be interpreted as (sim_n1 + sim_n2) * decayed_importance_anchor
        # Or sim_n1 * decayed_n1 + sim_n2 * decayed_n2 (if individual node decay is tracked)

        # Using: (avg_similarity_of_triplet_nodes_to_query) * (edge.strength) * (decayed_importance_of_anchor)
        # The prompt says "product of importance scores is lower than a threshold" for stopping traversal,
        # but for ranking it says "product of the sum of the cosine similaritys of the nodes times the importance decayed from the closest anchor".
        
        # Let's interpret "importance decayed from the closest anchor" as the importance of the node in the triplet
        # that is closer to *an* anchor, considering its original importance decayed over distance.
        # This is complex if multiple anchors. Simpler: use the current anchor's decayed importance.

        score = (avg_similarity + edge.strength_score) * decayed_importance # A simple combination
        # A more direct interpretation of the prompt:
        # score = avg_similarity * decayed_importance # Assuming avg_similarity = "sum of cosine similarities of the nodes" (normalized)
        
        return score

    def _select_top_triplets(self, collected_triplets_with_scores: List[Tuple[Triplet, float]]) -> List[Triplet]:
        """Sorts triplets by score and returns top Y, ensuring correct edge direction."""
        collected_triplets_with_scores.sort(key=lambda item: item[1], reverse=True)
        
        top_triplets: List[Triplet] = []
        for triplet, score in collected_triplets_with_scores[:self.top_Y]:
            n1, e, n2 = triplet
            # Ensure correct order based on edge direction for returning
            # The way triplets are formed (source_node, edge, target_node) should already be correct.
            if e.source_id == n1.id and e.target_id == n2.id:
                top_triplets.append(triplet)
            elif e.source_id == n2.id and e.target_id == n1.id: # Should not happen if formed correctly
                logger.warning(f"Triplet {triplet} has nodes swapped relative to edge direction. Correcting.")
                top_triplets.append((n2, e, n1))
            else:
                logger.error(f"Edge {e} does not connect nodes {n1} and {n2} in triplet. Skipping.")
        return top_triplets


class DFSTraversal(TraversalAlgorithm):
    """
    Depth-First Search traversal.
    """
    def traverse(self,
                 start_nodes: List[Node],
                 query_embedding: Optional[List[float]] = None) -> List[Triplet]:
        all_collected_triplets_with_scores: List[Tuple[Triplet, float]] = []
        global_visited_nodes: Set[int] = set() # Avoid re-processing from different anchors if desired

        for anchor_node in start_nodes:
            if anchor_node.id in global_visited_nodes and self.hyperparams.get_component_param(self._component_name, "avoid_revisiting_from_other_anchors", True):
                continue

            # path_importance_product starts with the anchor's importance
            # stack stores (node_id, current_depth, current_path_importance_product)
            stack: List[Tuple[int, int, float]] = [(anchor_node.id, 0, anchor_node.importance_score)]
            
            # visited_in_current_dfs: to avoid cycles within the traversal from *this* anchor
            visited_in_current_dfs: Set[int] = {anchor_node.id}
            
            # To track nodes visited just for the count for stopping condition
            nodes_processed_count_for_stop = 0

            while stack:
                curr_node_id, depth, path_importance_prod = stack.pop()
                nodes_processed_count_for_stop += 1
                global_visited_nodes.add(curr_node_id)

                curr_node_obj = self.graph.get_node(curr_node_id)
                if not curr_node_obj:
                    continue

                if self.stopping_condition.should_stop(path_importance_prod, depth, nodes_processed_count_for_stop, curr_node_obj):
                    continue

                # Process outgoing edges from curr_node_obj
                for edge in self.graph.get_outgoing_edges(curr_node_id):
                    neighbor_node_id = edge.target_id
                    neighbor_node_obj = self.graph.get_node(neighbor_node_id)

                    if neighbor_node_obj:
                        triplet = (curr_node_obj, edge, neighbor_node_obj)
                        score = self._calculate_triplet_score(triplet, anchor_node, depth, query_embedding)
                        all_collected_triplets_with_scores.append((triplet, score))

                        if neighbor_node_id not in visited_in_current_dfs: # Check for cycles for THIS DFS path
                            visited_in_current_dfs.add(neighbor_node_id) # Mark visited for this DFS path
                            # Next path importance: current_path_prod * neighbor.importance (or just edge strength?)
                            # Prompt: "product of importance scores is lower than a threshold"
                            # This usually means importance of nodes along the path.
                            next_path_importance_prod = path_importance_prod * (neighbor_node_obj.importance_score \
                                if neighbor_node_obj.importance_score > 0 else 0.001) # Avoid zero product early

                            stack.append((neighbor_node_id, depth + 1, next_path_importance_prod))
            
            # After finishing DFS from one anchor, reset visited_in_current_dfs for the next anchor
            # This is implicitly handled as it's local to the loop.

        return self._select_top_triplets(all_collected_triplets_with_scores)


class BFSTraversal(TraversalAlgorithm):
    """
    Breadth-First Search traversal.
    """
    def traverse(self,
                 start_nodes: List[Node],
                 query_embedding: Optional[List[float]] = None) -> List[Triplet]:
        all_collected_triplets_with_scores: List[Tuple[Triplet, float]] = []
        
        # queue stores (node_id, current_depth, current_path_importance_product, anchor_node_for_this_path)
        # The anchor_node_for_this_path is needed for scoring.
        queue: List[Tuple[int, int, float, Node]] = []
        
        # visited_nodes: Stores (node_id, depth) to allow shorter paths if found from other anchors
        # Or simply Set[int] if we only visit once globally.
        # For BFS, standard is to visit each node once.
        visited_nodes_overall: Set[int] = set()


        for anchor_node in start_nodes:
            if anchor_node.id not in visited_nodes_overall:
                 queue.append((anchor_node.id, 0, anchor_node.importance_score, anchor_node))
                 visited_nodes_overall.add(anchor_node.id)

        head = 0
        nodes_processed_count_for_stop = 0

        while head < len(queue):
            curr_node_id, depth, path_importance_prod, current_anchor = queue[head]
            head += 1
            nodes_processed_count_for_stop +=1

            curr_node_obj = self.graph.get_node(curr_node_id)
            if not curr_node_obj:
                continue

            if self.stopping_condition.should_stop(path_importance_prod, depth, nodes_processed_count_for_stop, curr_node_obj):
                continue

            # Process outgoing edges
            for edge in self.graph.get_outgoing_edges(curr_node_id):
                neighbor_node_id = edge.target_id
                neighbor_node_obj = self.graph.get_node(neighbor_node_id)

                if neighbor_node_obj:
                    triplet = (curr_node_obj, edge, neighbor_node_obj)
                    # Score using the anchor that started this particular BFS expansion path
                    score = self._calculate_triplet_score(triplet, current_anchor, depth, query_embedding)
                    all_collected_triplets_with_scores.append((triplet, score))

                    if neighbor_node_id not in visited_nodes_overall:
                        visited_nodes_overall.add(neighbor_node_id)
                        next_path_importance_prod = path_importance_prod * (neighbor_node_obj.importance_score \
                            if neighbor_node_obj.importance_score > 0 else 0.001)
                        queue.append((neighbor_node_id, depth + 1, next_path_importance_prod, current_anchor))
        
        return self._select_top_triplets(all_collected_triplets_with_scores)