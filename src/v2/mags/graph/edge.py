"""
Defines the Edge (Connection) structure for the Liquid Knowledge Graph.
"""
from typing import Any, Optional, Dict
from mags.hyperparameters.hyperparameters import Hyperparameters
from mags.graph.node import Node # For type hinting

class Edge:
    """
    Represents a directed connection between two nodes in the Liquid Knowledge Graph.
    Edges have content (label) and strength.
    """
    _id_counter = 0

    def __init__(self,
                 source_node_id: int,
                 target_node_id: int,
                 hyperparams: Hyperparameters,
                 edge_id: Optional[int] = None,
                 content: str = "", # e.g., "is_a", "related_to"
                 strength_score: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes an Edge.

        Args:
            source_node_id: ID of the source node.
            target_node_id: ID of the target node.
            hyperparams: Hyperparameters instance.
            edge_id: Optional ID. If None, a new unique ID is generated.
            content: Textual content or label of the edge (describes the relationship).
            strength_score: Initial strength score of the connection.
            metadata: Additional arbitrary data associated with the edge.
        """
        self.hyperparams = hyperparams

        if edge_id is None:
            self.id: int = Edge._generate_id()
        else:
            self.id: int = edge_id
            Edge._id_counter = max(Edge._id_counter, edge_id) # Ensure counter is ahead

        self.source_id: int = source_node_id
        self.target_id: int = target_node_id
        self.content: str = content
        self.strength_score: float = strength_score if strength_score is not None \
            else self.hyperparams.get_component_param("edge", "default_strength", 0.5)
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

    @classmethod
    def _generate_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset_id_counter(cls, value: int = 0) -> None:
        """Resets the global ID counter, primarily for testing."""
        cls._id_counter = value

    def update_strength(self, new_score: float) -> None:
        """Updates the strength score of the edge."""
        self.strength_score = new_score

    def __repr__(self) -> str:
        return (f"Edge(id={self.id}, {self.source_id} --[{self.content}, {self.strength_score:.2f}]--> {self.target_id})")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

# Register default hyperparameters for Edge
# Example:
# global_hyperparameters.register_defaults("edge", {
# "default_strength": 0.5
# })