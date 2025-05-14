"""
Defines the BaseGraph abstract class.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from mags.graph.node import Node
from mags.graph.edge import Edge
from mags.hyperparameters.hyperparameters import Hyperparameters

class BaseGraph(ABC):
    """
    Abstract base class for graph structures.
    """
    def __init__(self, hyperparams: Hyperparameters):
        self.hyperparams = hyperparams
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}
        # Adjacency list: node_id -> set of (neighbor_node_id, edge_id)
        # For directed graph, this stores outgoing edges.
        self.adj: Dict[int, List[Tuple[int, int]]] = {}
        # Reverse adjacency list for incoming edges, useful for some traversals or analysis
        self.rev_adj: Dict[int, List[Tuple[int, int]]] = {}


    @abstractmethod
    def add_node(self, node: Node) -> bool:
        """Adds a node to the graph."""
        pass

    @abstractmethod
    def get_node(self, node_id: int) -> Optional[Node]:
        """Retrieves a node by its ID."""
        pass

    @abstractmethod
    def remove_node(self, node_id: int) -> bool:
        """Removes a node and its incident edges from the graph."""
        pass

    @abstractmethod
    def add_edge(self, edge: Edge) -> bool:
        """Adds an edge to the graph."""
        pass

    @abstractmethod
    def get_edge(self, edge_id: int) -> Optional[Edge]:
        """Retrieves an edge by its ID."""
        pass

    @abstractmethod
    def remove_edge(self, edge_id: int) -> bool:
        """Removes an edge from the graph."""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: int) -> List[Node]:
        """Gets all neighbors of a given node."""
        pass

    @abstractmethod
    def get_outgoing_edges(self, node_id: int) -> List[Edge]:
        """Gets all outgoing edges from a given node."""
        pass

    @abstractmethod
    def get_incoming_edges(self, node_id: int) -> List[Edge]:
        """Gets all incoming edges to a given node."""
        pass

    @abstractmethod
    def get_node_count(self) -> int:
        """Returns the number of nodes in the graph."""
        pass

    @abstractmethod
    def get_edge_count(self) -> int:
        """Returns the number of edges in the graph."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Removes all nodes and edges from the graph."""
        pass

    def get_all_nodes(self) -> List[Node]:
        """Returns a list of all nodes in the graph."""
        return list(self.nodes.values())

    def get_all_edges(self) -> List[Edge]:
        """Returns a list of all edges in the graph."""
        return list(self.edges.values())

    def get_edge_between_nodes(self, source_id: int, target_id: int) -> Optional[Edge]:
        """Finds an edge between two nodes, if one exists."""
        if source_id in self.adj:
            for neighbor_id, edge_id in self.adj[source_id]:
                if neighbor_id == target_id:
                    return self.edges.get(edge_id)
        return None