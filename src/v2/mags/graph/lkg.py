"""
Liquid Knowledge Graph (LKG) implementation.
"""
import logging
from enum import Enum
from typing import Any, List, Optional, Tuple, Dict, Callable, Set
from mags.graph.base_graph import BaseGraph
from mags.graph.node import Node
from mags.graph.edge import Edge
from mags.hyperparameters.hyperparameters import Hyperparameters
from mags.utils.text_processing import are_texts_equivalent, normalize_text
# Import Modifier type hint when it's defined
# from mags.modifiers.base_modifier import BaseModifier, ModifierRunType

logger = logging.getLogger(__name__)

class ModifierPhase(Enum):
    BEFORE_QUERY = "before_query"
    AFTER_QUERY = "after_query"
    DETACHED = "detached"


class LiquidKnowledgeGraph(BaseGraph):
    """
    Liquid Knowledge Graph (LKG) stores memories (nodes) and connections (edges).
    It supports dynamic modifications and RAG-style queries.
    """
    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams)
        self._initialize_hyperparams()
        # Modifiers: phase -> list of (modifier_instance, name_if_detached)
        self.modifiers: Dict[ModifierPhase, List[Tuple[Any, Optional[str]]]] = {
            ModifierPhase.BEFORE_QUERY: [],
            ModifierPhase.AFTER_QUERY: [],
            ModifierPhase.DETACHED: []
        }
        # To quickly find detached modifiers by name
        self.detached_modifier_registry: Dict[str, Any] = {}
        self.change_count = 0 # For error calculation

    def _initialize_hyperparams(self):
        """Initialize LKG specific hyperparameters with defaults if not set."""
        defaults = {
            "max_nodes": 10000,
            "max_edges": 50000,
            "default_node_importance": 1.0,
            "default_edge_strength": 0.5,
            "allow_duplicate_node_content": False # If false, uses existing node on match
        }
        self.hyperparams.register_defaults("lkg", defaults)
        Node.reset_id_counter() # Reset ID counters when a new graph is made
        Edge.reset_id_counter()

    def add_node(self,
                 content: str,
                 importance_score: Optional[float] = None,
                 surprise_score: Optional[float] = None,
                 embeddings: Optional[List[float]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 is_permanent: bool = False) -> Node:
        """
        Adds a new node to the LKG.
        If a node with similar content already exists (based on normalization
        and hyperparameter `lkg.allow_duplicate_node_content`),
        it may return the existing node instead of creating a new one.

        Returns:
            The added or existing Node.
        """
        if not self.hyperparams.get_component_param("lkg", "allow_duplicate_node_content"):
            normalized_content = normalize_text(content)
            for existing_node in self.nodes.values():
                if are_texts_equivalent(existing_node.content, normalized_content):
                    logger.info(f"Node with similar content '{content}' found (ID: {existing_node.id}). Using existing node.")
                    return existing_node

        node = Node(content=content,
                    hyperparams=self.hyperparams,
                    importance_score=importance_score,
                    surprise_score=surprise_score,
                    embeddings=embeddings,
                    metadata=metadata,
                    is_permanent=is_permanent)

        if node.id in self.nodes:
            logger.warning(f"Node with ID {node.id} already exists. This might indicate an ID counter issue.")
            return self.nodes[node.id] # Should not happen if IDs are unique

        self.nodes[node.id] = node
        self.adj[node.id] = []
        self.rev_adj[node.id] = []
        self.change_count += 1
        logger.debug(f"Added node: {node}")
        return node

    def get_node(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)

    def find_node_by_content(self, content: str) -> Optional[Node]:
        """Finds a node by its normalized content."""
        normalized_content = normalize_text(content)
        for node in self.nodes.values():
            if are_texts_equivalent(node.content, normalized_content):
                return node
        return None

    def remove_node(self, node_id: int) -> bool:
        if node_id not in self.nodes:
            return False

        # Remove incident edges
        edges_to_remove = []
        if node_id in self.adj:
            for _, edge_id in self.adj[node_id]:
                edges_to_remove.append(edge_id)
        if node_id in self.rev_adj:
            for _, edge_id in self.rev_adj[node_id]:
                edges_to_remove.append(edge_id)

        for edge_id in set(edges_to_remove): # Use set to avoid duplicates
            self.remove_edge(edge_id) # This will also update adj/rev_adj

        del self.nodes[node_id]
        if node_id in self.adj: del self.adj[node_id]
        if node_id in self.rev_adj: del self.rev_adj[node_id]

        self.change_count += 1
        logger.debug(f"Removed node ID: {node_id}")
        return True

    def add_edge(self,
                 source_node_id: int,
                 target_node_id: int,
                 content: str = "",
                 strength_score: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[Edge]:
        """
        Adds a new directed edge to the LKG.
        Nodes must exist.
        """
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            logger.error(f"Cannot add edge: Source ({source_node_id}) or target ({target_node_id}) node does not exist.")
            return None

        edge = Edge(source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    hyperparams=self.hyperparams,
                    content=content,
                    strength_score=strength_score,
                    metadata=metadata)

        if edge.id in self.edges:
            logger.warning(f"Edge with ID {edge.id} already exists. This might indicate an ID counter issue.")
            return self.edges.get(edge.id) # Should not happen

        self.edges[edge.id] = edge
        self.adj.setdefault(source_node_id, []).append((target_node_id, edge.id))
        self.rev_adj.setdefault(target_node_id, []).append((source_node_id, edge.id))
        self.change_count += 1
        logger.debug(f"Added edge: {edge}")
        return edge

    def get_edge(self, edge_id: int) -> Optional[Edge]:
        return self.edges.get(edge_id)

    def remove_edge(self, edge_id: int) -> bool:
        edge = self.edges.pop(edge_id, None)
        if not edge:
            return False

        if edge.source_id in self.adj:
            self.adj[edge.source_id] = [(n_id, e_id) for n_id, e_id in self.adj[edge.source_id] if e_id != edge_id]
        if edge.target_id in self.rev_adj:
            self.rev_adj[edge.target_id] = [(n_id, e_id) for n_id, e_id in self.rev_adj[edge.target_id] if e_id != edge_id]

        self.change_count += 1
        logger.debug(f"Removed edge ID: {edge_id}")
        return True

    def add_triplet(self,
                    node1_content: str,
                    edge_content: str,
                    node2_content: str,
                    node1_params: Optional[Dict] = None, # e.g. importance, embeddings
                    node2_params: Optional[Dict] = None,
                    edge_params: Optional[Dict] = None   # e.g. strength
                   ) -> Tuple[Optional[Node], Optional[Edge], Optional[Node]]:
        """
        Adds a triplet (node1, edge, node2) to the graph.
        If nodes with similar content exist, they are reused.
        The edge direction is node1 -> node2.

        Returns:
            A tuple (node1, edge, node2). Elements can be None if creation failed.
        """
        node1_params = node1_params or {}
        node2_params = node2_params or {}
        edge_params = edge_params or {}

        # Find or create node1
        node1 = self.find_node_by_content(node1_content)
        if not node1:
            node1 = self.add_node(content=node1_content, **node1_params)
        elif node1_params: # Update existing node if params are provided
            if 'importance_score' in node1_params: node1.update_importance(node1_params['importance_score'])
            if 'embeddings' in node1_params: node1.embeddings = node1_params['embeddings'] # Direct update, or method

        # Find or create node2
        node2 = self.find_node_by_content(node2_content)
        if not node2:
            node2 = self.add_node(content=node2_content, **node2_params)
        elif node2_params:
            if 'importance_score' in node2_params: node2.update_importance(node2_params['importance_score'])
            if 'embeddings' in node2_params: node2.embeddings = node2_params['embeddings']

        if not node1 or not node2:
            logger.error("Failed to create/find nodes for the triplet.")
            return None, None, None

        # Add edge
        edge = self.add_edge(source_node_id=node1.id,
                             target_node_id=node2.id,
                             content=edge_content,
                             **edge_params)
        return node1, edge, node2

    def get_neighbors(self, node_id: int) -> List[Node]:
        if node_id not in self.nodes:
            return []
        neighbor_ids = set()
        if node_id in self.adj:
            neighbor_ids.update(n_id for n_id, _ in self.adj[node_id])
        if node_id in self.rev_adj: # If considering undirected neighbors
             neighbor_ids.update(n_id for n_id, _ in self.rev_adj[node_id])
        return [self.nodes[n_id] for n_id in neighbor_ids if n_id in self.nodes]


    def get_outgoing_edges(self, node_id: int) -> List[Edge]:
        if node_id not in self.adj:
            return []
        return [self.edges[edge_id] for _, edge_id in self.adj[node_id] if edge_id in self.edges]

    def get_incoming_edges(self, node_id: int) -> List[Edge]:
        if node_id not in self.rev_adj:
            return []
        return [self.edges[edge_id] for _, edge_id in self.rev_adj[node_id] if edge_id in self.edges]

    def get_node_count(self) -> int:
        return len(self.nodes)

    def get_edge_count(self) -> int:
        return len(self.edges)

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.adj.clear()
        self.rev_adj.clear()
        Node.reset_id_counter()
        Edge.reset_id_counter()
        self.change_count = 0
        logger.info("LKG cleared.")

    def graph_representation(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the graph (e.g., for serialization or visualization)."""
        return {
            "nodes": [vars(node) for node in self.nodes.values()],
            "edges": [vars(edge) for edge in self.edges.values()]
        }

    def register_modifier(self, modifier_instance: Any, phase: ModifierPhase, name: Optional[str] = None) -> None:
        """
        Registers a modifier to be run at a specific phase.
        If phase is DETACHED, a unique name must be provided.
        """
        if phase == ModifierPhase.DETACHED:
            if not name:
                raise ValueError("A name must be provided for DETACHED modifiers.")
            if name in self.detached_modifier_registry:
                raise ValueError(f"A DETACHED modifier with name '{name}' already exists.")
            self.detached_modifier_registry[name] = modifier_instance
        self.modifiers[phase].append((modifier_instance, name))
        logger.info(f"Registered modifier '{name if name else modifier_instance.__class__.__name__}' for phase {phase.value}")

    def run_modifiers(self, phase: ModifierPhase, *modifier_args: Tuple[Any, ...], detached_modifier_name: Optional[str] = None) -> None:
        """
        Runs all registered modifiers for a given phase or a specific detached modifier by name.

        For BEFORE_QUERY and AFTER_QUERY:
            Pass arguments for the modifiers as tuples.
            Example: run_modifiers(ModifierPhase.AFTER_QUERY, (arg1_mod1, arg2_mod1), (arg1_mod2,))
            The LKG (self) is automatically passed as the first argument to each modifier.
            If specific IDs are meant to be passed, they should be part of `modifier_args`.

        For DETACHED:
            Provide `detached_modifier_name` and its arguments in `modifier_args`.
            Example: run_modifiers(ModifierPhase.DETACHED, (arg1_detached_mod,), detached_modifier_name="my_decay")
        """
        if phase == ModifierPhase.DETACHED:
            if not detached_modifier_name:
                logger.error("Detached modifier name not provided.")
                return
            modifier = self.detached_modifier_registry.get(detached_modifier_name)
            if not modifier:
                logger.error(f"No detached modifier named '{detached_modifier_name}' found.")
                return
            try:
                # Detached modifiers might have different signatures, often (graph, *args)
                # Or, if they follow a strict (ids, graph, *args) then `ids` needs to be handled.
                # For simplicity, assume (graph, *args) or they handle args themselves
                logger.info(f"Running DETACHED modifier '{detached_modifier_name}' with args: {modifier_args}")
                if hasattr(modifier, 'apply'): # Convention: modifier has an 'apply' method
                     modifier.apply(self, *modifier_args[0] if modifier_args else [])
                else: # Fallback if it's a callable
                     modifier(self, *modifier_args[0] if modifier_args else [])
            except Exception as e:
                logger.error(f"Error running DETACHED modifier '{detached_modifier_name}': {e}", exc_info=True)
        else:
            # For BEFORE/AFTER, pass LKG as first arg, then unpack specific args for each modifier
            # The original request implied IDs are automatically passed as first arg(s).
            # This is tricky if different modifiers expect different ID structures.
            # A common convention is to pass the graph itself, and modifiers extract what they need.
            # Or, the calling context (MemoryBlock) should prepare specific IDs if needed.
            # For now, let's assume modifiers take (graph, *specific_args_for_that_modifier).
            # The `modifier_args` here would be a list of tuples, where each tuple is args for ONE modifier.

            if len(self.modifiers[phase]) != len(modifier_args) and modifier_args:
                 logger.warning(f"Mismatch between number of {phase.value} modifiers ({len(self.modifiers[phase])}) and provided argument sets ({len(modifier_args)}). This may lead to errors.")

            for i, (modifier_instance, _) in enumerate(self.modifiers[phase]):
                current_modifier_args = modifier_args[i] if i < len(modifier_args) else ()
                try:
                    logger.info(f"Running {phase.value} modifier {modifier_instance.__class__.__name__} with args: {current_modifier_args}")
                    if hasattr(modifier_instance, 'apply'):
                        # Pass the LKG instance as the first argument
                        modifier_instance.apply(self, *current_modifier_args)
                    else: # Fallback if it's a simple function (less likely with class-based modifiers)
                        modifier_instance(self, *current_modifier_args)
                except Exception as e:
                    logger.error(f"Error running {phase.value} modifier {modifier_instance.__class__.__name__}: {e}", exc_info=True)

    def __repr__(self) -> str:
        return f"<LiquidKnowledgeGraph nodes={self.get_node_count()}, edges={self.get_edge_count()}>"

    def get_all_triplets(self) -> List[Dict[str, Any]]:
        """
        Retrieves all triplets in the form:
        [
            {
                "id": edge_id,
                "triplet": [source_node_content, edge_content, target_node_content]
            },
            ...
        ]
        The result is ordered by edge ID.
        """
        triplets = []
        
        # Sort edges by ID to ensure ordering
        for edge_id in sorted(self.edges.keys()):
            edge = self.edges[edge_id]
            source_node = self.nodes.get(edge.source_id)
            target_node = self.nodes.get(edge.target_id)

            # Defensive check in case nodes were removed but edge wasn't cleaned up
            if source_node and target_node:
                triplets.append({
                    "id": edge_id,
                    "triplet": [source_node.content, edge.content, target_node.content]
                })
            else:
                logger.warning(f"Edge {edge_id} references missing nodes: "
                            f"{edge.source_id} or {edge.target_id}")
        
        return triplets