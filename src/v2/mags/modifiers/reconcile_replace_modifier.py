"""
Reconcile/Replace Modifier: Updates the graph based on a new triplet,
handling conflicts or overlaps.
"""
import logging
from typing import Tuple, Optional, Dict, Any
from mags.modifiers.base_modifier import BaseModifier, ModifierRunType
from mags.graph.lkg import LiquidKnowledgeGraph, Node, Edge # Assuming direct import for Node, Edge
from mags.utils.text_processing import are_texts_equivalent, normalize_text
from mags.hyperparameters.hyperparameters import Hyperparameters

logger = logging.getLogger(__name__)

class ReconcileReplaceModifier(BaseModifier):
    """
    Reconciles a new triplet (node1_content, edge_content, node2_content) with the graph.
    - If nodes are the same, it might replace the edge and reset scores.
    - If node1 and edge are the same, it might make the edge point to a new node2.
    - If edge is same, it might connect node1 and node2, removing the old edge's connections.
    On reconcile, scores of *changed* items are reset to defaults.
    """

    def __init__(self, hyperparams: Hyperparameters):
        super().__init__(hyperparams, run_type=ModifierRunType.DETACHED)
        self.hyperparams.register_defaults(
            self._component_name, # "reconcilereplace"
            {
                "reset_importance_on_reconcile": True,
                "reset_strength_on_reconcile": True,
                # Default scores are taken from Node/Edge classes via graph.hyperparams
            }
        )

    def _reset_node_scores(self, node: Node):
        if self._get_param("reset_importance_on_reconcile"):
            node.importance_score = node.hyperparams.get_component_param("node", "default_importance", 1.0)
            node.surprise_score = node.hyperparams.get_component_param("node", "default_surprise", 0.5)
            # metadata could also be reset or merged

    def _reset_edge_scores(self, edge: Edge):
        if self._get_param("reset_strength_on_reconcile"):
            edge.strength_score = edge.hyperparams.get_component_param("edge", "default_strength", 0.5)
            # metadata could also be reset

    def apply(self,
              graph: LiquidKnowledgeGraph,
              triplet: Tuple[str, str, str], # (node1_content, edge_content, node2_content)
              node1_new_params: Optional[Dict[str, Any]] = None,
              node2_new_params: Optional[Dict[str, Any]] = None,
              edge_new_params: Optional[Dict[str, Any]] = None,
              *args, **kwargs) -> Tuple[Optional[Node], Optional[Edge], Optional[Node]]:
        """
        Reconciles a new triplet with the graph.

        Args:
            graph: The LiquidKnowledgeGraph instance.
            triplet: A tuple (node1_content, edge_content, node2_content).
            node1_new_params: Optional new parameters for node1 if it's created/updated.
            node2_new_params: Optional new parameters for node2.
            edge_new_params: Optional new parameters for the edge.

        Returns:
            A tuple (node1, edge, node2) representing the state after reconciliation.
        """
        n1_content, e_content, n2_content = triplet
        n1_params = node1_new_params or {}
        n2_params = node2_new_params or {}
        e_params = edge_new_params or {}

        logger.info(f"Reconciling triplet: ('{n1_content}', '{e_content}', '{n2_content}')")

        # --- Find existing nodes ---
        # We use find_node_by_content which uses normalized comparison
        existing_n1 = graph.find_node_by_content(n1_content)
        existing_n2 = graph.find_node_by_content(n2_content)

        final_n1, final_e, final_n2 = None, None, None
        changed = False

        # --- Case 1: Nodes are the "same" (content match), focus on edge ---
        # "Same" means their content is equivalent.
        if existing_n1 and existing_n2 and existing_n1.id == existing_n2.id and are_texts_equivalent(n1_content, n2_content):
            # This is a self-loop or redundant info about a single entity.
            # If n1_content and n2_content are identical, we are effectively defining a property
            # of n1_content using e_content. Example: "Paris" - "is a city" - "Paris"
            # This interpretation might be better handled by ensuring n2 is distinct or by specific schema.
            # For now, if n1 and n2 are the *exact same node*, what to do with the edge?
            logger.warning(f"Reconcile: Node1 and Node2 refer to the same entity: {n1_content}. Behavior for self-edges with new content needs careful definition.")
            # Let's assume we add/update the self-loop if e_content is meaningful.
            final_n1 = existing_n1
            final_n2 = existing_n2 # same as final_n1

            # Check for existing edge with same content between these identical nodes
            existing_self_edge = None
            for _, edge_id_val in graph.adj.get(final_n1.id, []):
                edge_obj = graph.get_edge(edge_id_val)
                if edge_obj and edge_obj.target_id == final_n2.id and are_texts_equivalent(edge_obj.content, e_content):
                    existing_self_edge = edge_obj
                    break
            
            if existing_self_edge:
                final_e = existing_self_edge
                # Update if new params are different
                if e_params.get("strength_score", final_e.strength_score) != final_e.strength_score or \
                   e_params.get("content", final_e.content) != final_e.content: # Content update useful?
                    final_e.content = e_params.get("content", final_e.content)
                    self._reset_edge_scores(final_e) # Reset if content changed or explicitly asked
                    changed = True
            else:
                # Add new self-loop edge
                final_e = graph.add_edge(final_n1.id, final_n2.id, content=e_content, **e_params)
                if final_e: self._reset_edge_scores(final_e)
                changed = True
            return final_n1, final_e, final_n2


        # --- General Case: Nodes may or may not exist, edge may or may not exist ---
        
        # Get or create node1
        if existing_n1:
            final_n1 = existing_n1
            # Potentially update existing_n1 with n1_params if reconciliation implies update
            # For now, we assume if node exists, its core attributes aren't changed by reconcile, only scores
        else:
            final_n1 = graph.add_node(content=n1_content, **n1_params)
            if final_n1: self._reset_node_scores(final_n1)
            changed = True
        
        # Get or create node2
        if existing_n2:
            final_n2 = existing_n2
        else:
            final_n2 = graph.add_node(content=n2_content, **n2_params)
            if final_n2: self._reset_node_scores(final_n2)
            changed = True

        if not final_n1 or not final_n2:
            logger.error("Reconcile failed: Could not ensure both nodes exist.")
            return None, None, None

        # --- Handle the edge ---
        # Look for an existing edge between final_n1 and final_n2 with the *same content*
        # This is crucial for "if nodes are the same, it should replace the edge and reset score"
        # "replace" can mean: if edge content is different, remove old, add new.
        # If edge content is same, update existing.

        current_edge_between_n1_n2: Optional[Edge] = None
        for _, edge_id_val in graph.adj.get(final_n1.id, []):
            edge_obj = graph.get_edge(edge_id_val)
            if edge_obj and edge_obj.target_id == final_n2.id and are_texts_equivalent(edge_obj.content, e_content):
                current_edge_between_n1_n2 = edge_obj
                break
        
        if current_edge_between_n1_n2: # Edge with same content already exists
            final_e = current_edge_between_n1_n2
            # If new edge_params are provided (e.g. new strength), update it and reset.
            # For simplicity, if it exists, we'll reset its score as per "On reconcile, scores of changed items are reset."
            # Assuming 'changed item' can also be an existing item being re-asserted.
            self._reset_edge_scores(final_e)
            if "strength_score" in e_params: # Allow explicit override
                final_e.strength_score = e_params["strength_score"]

            changed = True # Even if just resetting scores of existing
        else:
            # No edge with this exact content exists between n1 and n2.
            # "if nodes are the same, it should replace the edge" -> this implies if an edge *of any content*
            # exists, it might be replaced. This is more complex. The prompt is a bit ambiguous.
            # Let's assume "replace the edge" means: if an edge *with different content* exists, remove it.
            # And then add the new one.
            
            # Simple interpretation: Add the new edge specified by e_content.
            # If there were other edges between final_n1 and final_n2 with *different* content, they remain.
            # This interpretation aligns with "If edge is same, it should connect node1 and node2, and get rid of the old edge." -
            # this seems to refer to finding an edge by content *elsewhere* and re-wiring it.

            # Let's refine: If there's an existing edge between n1 and n2, but its content is DIFFERENT
            # from e_content, the prompt says "replace the edge". This means remove the old, add the new.
            old_edges_to_remove_ids = []
            if final_n1.id in graph.adj:
                for neighbor_id, edge_id_val in graph.adj[final_n1.id]:
                    if neighbor_id == final_n2.id: # An edge exists from n1 to n2
                        existing_edge = graph.get_edge(edge_id_val)
                        if existing_edge and not are_texts_equivalent(existing_edge.content, e_content):
                            old_edges_to_remove_ids.append(existing_edge.id)
            
            for old_edge_id in old_edges_to_remove_ids:
                graph.remove_edge(old_edge_id)
                logger.info(f"Reconcile: Removed old edge ID {old_edge_id} between {final_n1.id} and {final_n2.id} to replace with new content.")
                changed = True

            # Now, add the new edge
            final_e = graph.add_edge(final_n1.id, final_n2.id, content=e_content, **e_params)
            if final_e:
                self._reset_edge_scores(final_e)
            changed = True


        # --- "If node1 and edge are the same, it should make edge point to node2 (and vice versa)" ---
        # This implies finding an existing edge (e_content) connected to an existing node (n1_content),
        # and if its other end is NOT n2_content, disconnect and reconnect to n2_content.
        # This is a more advanced re-wiring.
        # Let's assume `existing_n1` and `e_content` are the keys.
        if existing_n1:
            edges_from_n1_with_e_content: List[Edge] = []
            for _, edge_id_val in graph.adj.get(existing_n1.id, []):
                edge_obj = graph.get_edge(edge_id_val)
                if edge_obj and are_texts_equivalent(edge_obj.content, e_content):
                    edges_from_n1_with_e_content.append(edge_obj)
            
            for edge_to_rewire in edges_from_n1_with_e_content:
                if edge_to_rewire.target_id != final_n2.id: # Points to a different node
                    # Original target node
                    original_target_node = graph.get_node(edge_to_rewire.target_id)
                    logger.info(f"Reconcile: Edge '{edge_to_rewire.content}' (ID {edge_to_rewire.id}) from '{existing_n1.content}' currently points to '{original_target_node.content if original_target_node else 'Unknown'}'. Rewiring to '{final_n2.content}'.")
                    
                    # Remove from old target's rev_adj
                    if original_target_node and edge_to_rewire.id in [e_id for _, e_id in graph.rev_adj.get(original_target_node.id, [])]:
                         graph.rev_adj[original_target_node.id] = [(n, e_id) for n, e_id in graph.rev_adj[original_target_node.id] if e_id != edge_to_rewire.id]

                    edge_to_rewire.target_id = final_n2.id # Rewire
                    graph.rev_adj.setdefault(final_n2.id, []).append((existing_n1.id, edge_to_rewire.id)) # Update new target's rev_adj
                    self._reset_edge_scores(edge_to_rewire)
                    if final_e is None: final_e = edge_to_rewire # If we hadn't set final_e yet
                    changed = True
        # Vice-versa (if existing_n2 and e_content lead to an edge whose source is not final_n1)
        # This part is analogous and can be complex. For now, focusing on the n1->e case.

        # --- "If edge is same, it should connect node1 and node2, and get rid of the old edge." ---
        # This implies finding an edge by `e_content` *anywhere* in the graph,
        # then making its source `final_n1` and target `final_n2`.
        # This is a major operation and could mean taking an existing edge and completely re-assigning it.
        # This seems less common for "reconciliation" and more like "edge takeover".
        # For now, this interpretation is complex and might be too destructive without more context.
        # The earlier logic handles adding/updating the edge between the specific n1 and n2.

        if changed:
            graph.change_count +=1 # Increment once if any change occurred. Individual changes already counted.

        if not final_e and final_n1 and final_n2: # If no edge was found/created but nodes are there
             # This could happen if the logic above didn't explicitly create one
             # and the "rewiring" didn't apply. Ensure an edge exists.
             temp_e = graph.add_edge(final_n1.id, final_n2.id, content=e_content, **e_params)
             if temp_e:
                 self._reset_edge_scores(temp_e)
                 final_e = temp_e


        logger.info(f"Reconciliation result: N1: {final_n1}, E: {final_e}, N2: {final_n2}")
        if not args and not kwargs:
            pass
        return final_n1, final_e, final_n2