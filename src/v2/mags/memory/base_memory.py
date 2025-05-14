"""
Base Memory Block class.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Dict, Set
from mags.graph.lkg import LiquidKnowledgeGraph, ModifierPhase, Node, Edge
from mags.hyperparameters.hyperparameters import Hyperparameters
from mags.query.anchor_search import AnchorSearch # Specific types like MaximumAnchorSearch
from mags.query.traversal import TraversalAlgorithm, Triplet # Specific types like DFSTraversal
from mags.inference.Embedder import LocalEmbedder
# Placeholder for actual modifier classes, assuming they are registered with LKG
# from mags.modifiers.base_modifier import BaseModifier


logger = logging.getLogger(__name__)

class BaseMemoryBlock(ABC):
    """
    Parent class for different types of memory blocks (Semantic, Episodic, General).
    Each memory block contains a Liquid Knowledge Graph (LKG).
    """
    def __init__(self, memory_type: str, hyperparams: Hyperparameters):
        self.memory_type = memory_type
        self.hyperparams = hyperparams
        self.lkg = LiquidKnowledgeGraph(hyperparams)
        self._initialize_components()

        # Store last query results for potential use by AFTER_QUERY modifiers
        self._last_query_recalled_triplets: List[Triplet] = []
        self._last_query_recalled_node_ids: Set[int] = set()
        self._last_query_recalled_edge_ids: Set[int] = set()

        self.embed_function = LocalEmbedder()


    @abstractmethod
    def _initialize_components(self):
        """Initializes anchor search and traversal algorithms specific to this memory type."""
        self.anchor_search_strategy: Optional[AnchorSearch] = None
        self.traversal_strategy: Optional[TraversalAlgorithm] = None
        pass

    def add_modifier(self, modifier_instance: Any, phase: ModifierPhase, name: Optional[str] = None):
        """Registers a modifier with the underlying LKG."""
        self.lkg.register_modifier(modifier_instance, phase, name)
        logger.info(f"Modifier {modifier_instance.__class__.__name__} added to {self.memory_type} memory LKG for phase {phase}.")

    def run_lkg_modifiers(self, phase: ModifierPhase, *modifier_args: Tuple[Any, ...], detached_modifier_name: Optional[str] = None) -> None:
        """
        Client-facing method to run modifiers on the LKG.
        For DETACHED, pass detached_modifier_name and its args in modifier_args (as a single tuple).
        For BEFORE/AFTER, modifier_args is a sequence of tuples, each for one modifier.
        """
        self.lkg.run_modifiers(phase, *modifier_args, detached_modifier_name=detached_modifier_name)


    def query_memory(self, query_text: str, query_embedding: Optional[List[float]] = None, **kwargs) -> List[Triplet]:
        """
        Runs a query over the memory block and its LKG.

        The process:
        1. Run @before modifiers.
        2. Perform anchor search.
        3. Perform traversal from anchors.
        4. Collect and rank triplets.
        5. Run @after modifiers.
        6. Return results.
        """
        if not self.anchor_search_strategy or not self.traversal_strategy:
            logger.error(f"Anchor search or traversal strategy not initialized for {self.memory_type} memory.")
            return []

        logger.info(f"Querying {self.memory_type} memory with: '{query_text[:50]}...'")

        # 1. Run @before modifiers
        # Args for BEFORE modifiers need to be determined by the system design.
        # Example: might pass the query_text or query_embedding.
        # For now, assuming no specific args needed from client for common BEFORE mods.
        # If a BEFORE modifier needs specific args, the client of MemoryBlock should pass them.
        # Let's assume for now that `before_modifier_args` is empty or handled internally.
        before_modifier_args = kwargs.get("before_modifier_args", ())
        self.lkg.run_modifiers(ModifierPhase.BEFORE_QUERY, *before_modifier_args)


        # 2. Perform anchor search
        # Anchor search might use query_embedding or other kwargs
        anchor_nodes: List[Node] = self.anchor_search_strategy.find_anchors(
            query_embedding=query_embedding,
            **kwargs.get("anchor_search_kwargs", {})
        )
        if not anchor_nodes:
            logger.info("No anchor nodes found.")
            # Still run AFTER_QUERY modifiers even if no anchors found, as they might perform cleanup.
            after_modifier_args = kwargs.get("after_modifier_args", ()) # Prepare args for AFTER mods
            self.lkg.run_modifiers(ModifierPhase.AFTER_QUERY, *after_modifier_args)
            return []
        logger.debug(f"Found {len(anchor_nodes)} anchor(s): {[n.id for n in anchor_nodes]}")

        # 3. Perform traversal
        recalled_triplets: List[Triplet] = self.traversal_strategy.traverse(
            start_nodes=anchor_nodes,
            query_embedding=query_embedding
        )
        logger.debug(f"Traversal yielded {len(recalled_triplets)} triplets before ranking/selection.")

        # Store details for AFTER_QUERY modifiers
        self._last_query_recalled_triplets = recalled_triplets
        self._last_query_recalled_node_ids = set()
        self._last_query_recalled_edge_ids = set()
        for n1, e, n2 in recalled_triplets:
            self._last_query_recalled_node_ids.add(n1.id)
            self._last_query_recalled_node_ids.add(n2.id)
            self._last_query_recalled_edge_ids.add(e.id)

        # 5. Run @after modifiers
        # These modifiers might use the results of the query (e.g., recalled_triplets).
        # The prompt: "memory.run_modifiers(AFTER, (arg1,arg2,),(arg1,arg2,)...) is run from the client"
        # This implies the client prepares these args. The MemoryBlock could also prepare some default ones.
        # Example: if StrengthenModifier is registered, it needs recalled IDs.
        # Let's assume specific args are passed via `kwargs` or prepared if known.
        
        # Auto-pass recalled IDs to StrengthenModifier if present?
        # This is complex as `run_modifiers` takes a list of arg tuples.
        # A convention could be that AFTER modifiers requiring recalled_ids look for them
        # in `kwargs` passed to their `apply` method, or the LKG/MemoryBlock makes them available.
        # The current LKG.run_modifiers passes `self` (the LKG) and then `*current_modifier_args`.
        # Simplest: if an AFTER modifier is known to need these, the client passes them.
        # Or, the MemoryBlock's `run_lkg_modifiers` could be enhanced.
        
        # For now, rely on client passing args for AFTER mods as specified
        after_modifier_args = kwargs.get("after_modifier_args", ())
        # Example of how a client *might* prepare args for a StrengthenModifier:
        # after_args_for_strengthen = (self._last_query_recalled_node_ids, self._last_query_recalled_edge_ids)
        # client_calls: memory_block.run_lkg_modifiers(ModifierPhase.AFTER_QUERY, after_args_for_strengthen)
        # If multiple AFTER mods, it would be: ((ids_for_strength), (feedback_for_feedback_mod))
        self.lkg.run_modifiers(ModifierPhase.AFTER_QUERY, *after_modifier_args)


        # 6. Return results
        logger.info(f"Query completed. Returning {len(recalled_triplets)} triplets.")
        return recalled_triplets

    # Convenience methods to interact with LKG
    def add_node_to_lkg(self, content: str, **kwargs) -> Node:
        return self.lkg.add_node(content, **kwargs)

    def add_edge_to_lkg(self, source_node_id: int, target_node_id: int, **kwargs) -> Optional[Edge]:
        return self.lkg.add_edge(source_node_id, target_node_id, **kwargs)

    def add_triplet_to_lkg(self, node1_content: str, edge_content: str, node2_content: str, **kwargs) -> Tuple[Optional[Node], Optional[Edge], Optional[Node]]:
        embedding_1 = self.embed_function.embed_text(node1_content)
        embedding_2 = self.embed_function.embed_text(node2_content)

        node1_params = {
            "embeddings": embedding_1
        }

        node2_params = {
            "embeddings": embedding_2
        }

        return self.lkg.add_triplet(node1_content, edge_content, node2_content, node1_params=node1_params, node2_params=node2_params, **kwargs)

    def get_lkg_node_count(self) -> int:
        return self.lkg.get_node_count()

    def get_lkg_edge_count(self) -> int:
        return self.lkg.get_edge_count()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type='{self.memory_type}', LKG nodes={self.get_lkg_node_count()}, LKG edges={self.get_lkg_edge_count()}>"
    
    def get_all_triplets(self) -> List[Dict[str, Any]]:
        return self.lkg.get_all_triplets()