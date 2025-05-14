import time
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from Graph import Graph, Node, Edge

class HebPlasticity:
    """
    Implements Hebbian plasticity mechanisms for MAGS
    Handles edge weight updates based on co-activation patterns and contribution to responses
    """
    def __init__(self, 
                 base_delta: float = 0.1, 
                 decay_factor: float = 0.9,
                 decay_interval: float = 60.0,  # seconds
                 min_weight: float = 0.1,
                 max_weight: float = 5.0):
        """        
        Args:
            base_delta: Base amount to strengthen edges
            decay_factor: Factor for recency decay (0-1)
            decay_interval: Time interval for decay calculation (seconds)
            min_weight: Minimum edge weight
            max_weight: Maximum edge weight
        """
        self.base_delta = base_delta
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.active_paths = []  # paths used in current recall

    def record_path(self, path: List[str]):
        self.active_paths.append(path)
        
    def clear_paths(self):
        self.active_paths = []

    def strengthen_paths(self, graph: Graph, reward: float = 1.0):
        for path in self.active_paths:
            for i in range(len(path) - 1):
                source_id = path[i]
                target_id = path[i + 1]
                
                for edge in graph.edges:
                    if edge.source == source_id and edge.target == target_id:
                        edge.weight += reward * self.base_delta
                        edge.weight = min(edge.weight, self.max_weight)
                        break
    
        self.clear_paths()
    
    def apply_recency_decay(self, graph: Graph):
        current_time = time.time()
        for edge in graph.edges:
            if hasattr(edge, 'permanent') and edge.permanent:
                continue
            elapsed = current_time - getattr(edge, 'last_updated', current_time)
            decay = self.decay_factor ** (elapsed / self.decay_interval)
            edge.weight = max(self.min_weight, edge.weight * decay)
    
    def prune_weak_edges(self, graph: Graph, threshold: float = 0.2):
        """
        Returns: List of removed edge IDs
        """
        edges_to_remove = []
        for edge in graph.edges:
            if edge.weight < threshold:
                edges_to_remove.append(edge.id)
        for edge_id in edges_to_remove:
            graph.remove_edge(edge_id)
            
        return edges_to_remove