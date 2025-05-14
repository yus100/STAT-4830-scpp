"""
Defines the Node (Memory) structure for the Liquid Knowledge Graph.
"""
from typing import Any, List, Optional, Dict
from mags.hyperparameters.hyperparameters import Hyperparameters

class Node:
    """
    Represents a memory node in the Liquid Knowledge Graph.
    Nodes have content, importance, surprise, and embeddings.
    """
    _id_counter = 0

    def __init__(self,
                 content: str,
                 hyperparams: Hyperparameters,
                 node_id: Optional[int] = None,
                 importance_score: Optional[float] = None,
                 surprise_score: Optional[float] = None,
                 embeddings: Optional[List[float]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 is_permanent: bool = False):
        """
        Initializes a Node.

        Args:
            content: The textual content of the node.
            hyperparams: Hyperparameters instance.
            node_id: Optional ID. If None, a new unique ID is generated.
            importance_score: Initial importance score.
            surprise_score: Initial surprise score.
            embeddings: Vector representation of the node's content.
            metadata: Additional arbitrary data associated with the node.
            is_permanent: Flag indicating if the node is permanent.
        """
        self.hyperparams = hyperparams

        if node_id is None:
            self.id: int = Node._generate_id()
        else:
            self.id: int = node_id
            Node._id_counter = max(Node._id_counter, node_id) # Ensure counter is ahead

        self.content: str = content
        self.importance_score: float = importance_score if importance_score is not None \
            else self.hyperparams.get_component_param("node", "default_importance", 1.0)
        self.surprise_score: float = surprise_score if surprise_score is not None \
            else self.hyperparams.get_component_param("node", "default_surprise", 0.5)
        self.embeddings: List[float] = embeddings if embeddings is not None else []
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.is_permanent: bool = is_permanent

    @classmethod
    def _generate_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset_id_counter(cls, value: int = 0) -> None:
        """Resets the global ID counter, primarily for testing."""
        cls._id_counter = value

    def update_importance(self, new_score: float) -> None:
        """Updates the importance score of the node."""
        if not self.is_permanent: # Permanent nodes might have fixed importance or different update rules
            self.importance_score = new_score

    def update_surprise(self, new_score: float) -> None:
        """Updates the surprise score of the node."""
        self.surprise_score = new_score

    def set_permanent(self, status: bool = True) -> None:
        """Marks the node as permanent."""
        self.is_permanent = status

    def __repr__(self) -> str:
        return (f"Node(id={self.id}, content='{self.content[:30]}...', "
                f"importance={self.importance_score:.2f}, "
                f"permanent={self.is_permanent})")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

# Register default hyperparameters for Node
# This should ideally be done once when the Hyperparameters object is initialized
# For now, we assume it's handled at a higher level application setup.
# Example:
# global_hyperparameters.register_defaults("node", {
#     "default_importance": 1.0,
#     "default_surprise": 0.5
# })