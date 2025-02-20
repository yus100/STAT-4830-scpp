import time
import math
import requests
import networkx as nx
import numpy as np
from typing import List, Optional

# =============================================================================
# Base MemoryGraph Class
# =============================================================================
class MemoryGraph:
    def __init__(self, name: str = "MemoryGraph"):
        # Directed graph supports asymmetric edge weights.
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.name = name

    def add_node(self, content: str, **attributes) -> int:
        """
        Add a node with the given content.
        Default attributes include:
          - timestamp: time of insertion.
          - recency: a score (default 1.0) that decays over time.
          - importance: a score (initialized to 1.0).
        """
        node_id = self.node_counter
        self.graph.add_node(
            node_id,
            content=content,
            timestamp=time.time(),
            recency=1.0,
            importance=1.0,
            **attributes,
        )
        self.node_counter += 1
        return node_id

    def update_node(self, node_id: int, new_content: str, **attributes):
        """Update node content and refresh its timestamp and recency."""
        if node_id in self.graph:
            self.graph.nodes[node_id]["content"] = new_content
            self.graph.nodes[node_id]["timestamp"] = time.time()
            self.graph.nodes[node_id]["recency"] = 1.0  # Reset recency
            for key, value in attributes.items():
                self.graph.nodes[node_id][key] = value

    def add_edge(self, source: int, target: int, weight: float = 1.0):
        """Add an edge with a given weight and a timestamp."""
        self.graph.add_edge(source, target, weight=weight, last_updated=time.time())

    def update_edge(self, source: int, target: int, delta: float = 0.1):
        """
        Update an edge’s weight using a Hebbian-style increase.
        Also update the timestamp.
        """
        if self.graph.has_edge(source, target):
            self.graph[source][target]["weight"] += delta
            self.graph[source][target]["last_updated"] = time.time()

    def recall(self, prompt: str, similarity_threshold: float = 0.2) -> List[int]:
        """
        Retrieve node IDs whose content is similar to the prompt.
        Uses a simple Jaccard similarity (word overlap).
        """
        def similarity(a: str, b: str) -> float:
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            return len(set_a.intersection(set_b)) / max(len(set_a.union(set_b)), 1)

        recalled = []
        for node, data in self.graph.nodes(data=True):
            sim = similarity(prompt, data.get("content", ""))
            if sim >= similarity_threshold:
                recalled.append(node)
        return recalled

    def dfs_recall(
        self,
        start_node: int,
        weight_threshold: float = 0.1,
        path_product: float = 1.0,
        visited: Optional[set] = None,
    ) -> List[int]:
        """
        Depth-first search (DFS) starting from start_node.
        Traversal stops when the product of edge weights along a path falls below threshold.
        """
        if visited is None:
            visited = set()
        visited.add(start_node)
        results = [start_node]
        for neighbor in self.graph.successors(start_node):
            edge_weight = self.graph[start_node][neighbor].get("weight", 1.0)
            new_product = path_product * edge_weight
            if neighbor not in visited and new_product > weight_threshold:
                results.extend(self.dfs_recall(neighbor, weight_threshold, new_product, visited))
        return results

    def recency_decay(self, decay_factor: float = 0.9):
        """
        Apply a decay to each node's recency based on elapsed time.
        (Here we use a simple exponential decay per minute.)
        """
        current_time = time.time()
        for node, data in self.graph.nodes(data=True):
            elapsed = current_time - data.get("timestamp", current_time)
            decay = decay_factor ** (elapsed / 60.0)
            self.graph.nodes[node]["recency"] *= decay

    def compute_importance(self, node: int) -> float:
        """
        Compute an importance score for a node. For example:
          importance = (degree * recency)
        """
        degree = self.graph.degree(node)
        recency = self.graph.nodes[node].get("recency", 1.0)
        importance = degree * recency
        self.graph.nodes[node]["importance"] = importance
        return importance

    def consolidate(self, k: int = 2) -> List[int]:
        """
        Consolidate the memory using k-core decomposition.
        Nodes not in the k-core are pruned.
        Returns a list of removed node IDs.
        """
        try:
            core_nodes = set(nx.k_core(self.graph, k=k).nodes())
        except nx.NetworkXError:
            core_nodes = set()
        to_remove = [node for node in self.graph.nodes if node not in core_nodes]
        self.graph.remove_nodes_from(to_remove)
        return to_remove

    def get_all_nodes(self) -> List[int]:
        return list(self.graph.nodes)

    def add_graph_attention_edges(self, new_node: int, similarity_threshold: float = 0.3):
        """
        (Graph-Attention inspired) Check all existing nodes for content similarity with
        the new node’s content. If similarity exceeds threshold, add bidirectional edges
        with an initial low weight.
        """
        new_content = self.graph.nodes[new_node].get("content", "")
        def similarity(a: str, b: str) -> float:
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            return len(set_a.intersection(set_b)) / max(len(set_a.union(set_b)), 1)

        for node, data in self.graph.nodes(data=True):
            if node == new_node:
                continue
            sim = similarity(new_content, data.get("content", ""))
            if sim >= similarity_threshold and not self.graph.has_edge(new_node, node):
                self.add_edge(new_node, node, weight=0.3)
                self.add_edge(node, new_node, weight=0.3)


# =============================================================================
# Specialized Memory Blocks
# =============================================================================
class EpisodicMemoryBlock(MemoryGraph):
    def __init__(self):
        super().__init__(name="EpisodicMemoryBlock")

    def clear(self):
        """Clear all episodic memory (e.g., at session end)."""
        self.graph.clear()
        self.node_counter = 0


class SemanticMemoryBlock(MemoryGraph):
    def __init__(self):
        super().__init__(name="SemanticMemoryBlock")
    # Semantic memory persists between sessions and updates more slowly.


# =============================================================================
# RL Agent: A Simple Q-Learning Based Reinforcement Learner
# =============================================================================
class RLAgent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        """
        Initializes the RL agent.
          - learning_rate (α): how strongly new rewards update Q-values.
          - discount_factor (γ): not used explicitly here but reserved for future extensions.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = {}  # Dictionary mapping (source, target) -> Q-value

    def get_q(self, source: int, target: int) -> float:
        """Return the current Q-value for an edge (default 0.0)."""
        return self.Q.get((source, target), 0.0)

    def update_q(self, source: int, target: int, reward: float) -> float:
        """
        Update Q-value for the edge (source, target) using a simple Q-learning rule:
          Q_new = Q_old + α * (reward - Q_old)
        """
        current_q = self.get_q(source, target)
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.Q[(source, target)] = new_q
        return new_q

    def apply_reward(self, memory_graph: MemoryGraph, path: List[int], reward: float):
        """
        For each edge along the given DFS path, update its Q-value and
        adjust the memory graph edge weight accordingly.
        """
        for i in range(len(path) - 1):
            s = path[i]
            t = path[i + 1]
            new_q = self.update_q(s, t, reward)
            if memory_graph.graph.has_edge(s, t):
                memory_graph.graph[s][t]["weight"] = new_q


# =============================================================================
# LLM Interface (Using Ollama to Host a Local Llama-1.5B Instance)
# =============================================================================
class OllamaLLM:
    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "llama-1.5B"):
        self.endpoint = endpoint
        self.model = model

    def query(self, prompt: str) -> str:
        """
        Query the locally hosted LLM via Ollama.
        Adjust the payload as needed.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 100
        }
        try:
            response = requests.post(self.endpoint, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Exception: {str(e)}"


# =============================================================================
# MAGS: Memory Augmented Generative System
# =============================================================================
class MAGS:
    def __init__(
        self,
        llm: OllamaLLM,
        episodic: EpisodicMemoryBlock,
        semantic: SemanticMemoryBlock,
        agent: RLAgent,
    ):
        self.llm = llm
        self.episodic = episodic
        self.semantic = semantic
        self.agent = agent

    def process_input(self, prompt: str) -> str:
        """
        Process an input prompt by:
          1. Recalling from episodic and semantic memory.
          2. Querying the LLM with a combined memory context.
          3. Updating memory (engram step) in both blocks.
          4. Applying RL updates along DFS paths from recalled nodes.
          5. Applying recency decay and computing node importance.
        """
        # --- Recall Step ---
        recalled_epi = self.episodic.recall(prompt)
        recalled_sem = self.semantic.recall(prompt)
        recalled_nodes = list(set(recalled_epi + recalled_sem))
        recalled_context = []
        for node in recalled_nodes:
            if node in self.episodic.graph:
                recalled_context.append(f"[eMB]: {self.episodic.graph.nodes[node]['content']}")
            elif node in self.semantic.graph:
                recalled_context.append(f"[sMB]: {self.semantic.graph.nodes[node]['content']}")
        memory_context = "\n".join(recalled_context)
        full_prompt = f"{memory_context}\n{prompt}" if memory_context else prompt

        # --- Query the LLM ---
        response = self.llm.query(full_prompt)

        # --- Engram Step: Update Episodic Memory ---
        prompt_node_epi = self.episodic.add_node(prompt)
        response_node_epi = self.episodic.add_node(response)
        self.episodic.add_edge(prompt_node_epi, response_node_epi, weight=1.0)
        self.episodic.add_graph_attention_edges(prompt_node_epi)
        self.episodic.add_graph_attention_edges(response_node_epi)

        # --- Engram Step: Update Semantic Memory ---
        similar_prompt_sem = self.semantic.recall(prompt)
        if not similar_prompt_sem:
            prompt_node_sem = self.semantic.add_node(prompt)
        else:
            prompt_node_sem = similar_prompt_sem[0]
            self.semantic.update_node(prompt_node_sem, prompt)
        similar_response_sem = self.semantic.recall(response)
        if not similar_response_sem:
            response_node_sem = self.semantic.add_node(response)
        else:
            response_node_sem = similar_response_sem[0]
            self.semantic.update_node(response_node_sem, response)
        self.semantic.add_edge(prompt_node_sem, response_node_sem, weight=1.0)
        self.semantic.add_graph_attention_edges(prompt_node_sem)
        self.semantic.add_graph_attention_edges(response_node_sem)

        # --- RL Update: Apply reward along DFS paths in episodic memory ---
        # Here we simulate a reward signal (e.g., 0.5) for demonstration.
        dummy_reward = 0.5
        for recalled in recalled_nodes:
            path = self.episodic.dfs_recall(recalled, weight_threshold=0.1)
            if len(path) > 1:
                self.agent.apply_reward(self.episodic, path, reward=dummy_reward)

        # --- Update Recency and Importance ---
        self.episodic.recency_decay(decay_factor=0.98)
        self.semantic.recency_decay(decay_factor=0.99)
        for node in self.episodic.get_all_nodes():
            self.episodic.compute_importance(node)
        for node in self.semantic.get_all_nodes():
            self.semantic.compute_importance(node)

        return response

    def consolidate_memory(self) -> List[int]:
        """
        Consolidate episodic memory into semantic memory by transferring nodes
        not yet represented in semantic memory, and prune episodic memory.
        """
        nodes = self.episodic.get_all_nodes()
        for node in nodes:
            content = self.episodic.graph.nodes[node]["content"]
            similar = self.semantic.recall(content)
            if not similar:
                self.semantic.add_node(content)
        pruned_nodes = self.episodic.consolidate(k=1)
        return pruned_nodes


# =============================================================================
# Optional: A Simple Game Simulator for Testing Adaptation
# =============================================================================
class GameSimulator:
    """
    A simple game simulation that presents evolving rules to the MAGS system.
    The rules change over rounds; the model’s responses (and memory updates) are tracked.
    """
    def __init__(self, mags: MAGS):
        self.mags = mags
        self.round = 0
        self.rules = "Rule: Answer truthfully."

    def update_rules(self):
        # Change the rule every 3 rounds for demonstration.
        if self.round % 3 == 0:
            self.rules = f"Rule: In round {self.round}, answer with a twist!"

    def play_round(self, prompt: str) -> str:
        self.round += 1
        self.update_rules()
        full_prompt = f"{self.rules}\n{prompt}"
        response = self.mags.process_input(full_prompt)
        return response

    def run_simulation(self, rounds: int = 5):
        results = []
        for i in range(rounds):
            prompt = f"This is prompt number {i}."
            response = self.play_round(prompt)
            results.append((self.round, prompt, response))
        pruned = self.mags.consolidate_memory()
        return results, pruned


# =============================================================================
# Unit Tests for the Advanced MAGS with RL
# =============================================================================
if __name__ == "__main__":
    import unittest

    class TestMemoryGraph(unittest.TestCase):
        def setUp(self):
            self.mg = MemoryGraph("TestGraph")

        def test_add_and_update_node(self):
            node = self.mg.add_node("Initial content")
            self.assertIn(node, self.mg.graph.nodes)
            self.assertEqual(self.mg.graph.nodes[node]["content"], "Initial content")
            self.mg.update_node(node, "Updated content")
            self.assertEqual(self.mg.graph.nodes[node]["content"], "Updated content")

        def test_add_and_update_edge(self):
            n1 = self.mg.add_node("Node 1")
            n2 = self.mg.add_node("Node 2")
            self.mg.add_edge(n1, n2, weight=0.8)
            self.assertTrue(self.mg.graph.has_edge(n1, n2))
            init_weight = self.mg.graph[n1][n2]["weight"]
            self.mg.update_edge(n1, n2, delta=0.2)
            self.assertAlmostEqual(self.mg.graph[n1][n2]["weight"], init_weight + 0.2)

        def test_recall_and_dfs(self):
            n1 = self.mg.add_node("Memory about apples and fruit")
            n2 = self.mg.add_node("Memory about oranges and citrus")
            recalled = self.mg.recall("apples")
            self.assertIn(n1, recalled)
            self.assertNotIn(n2, recalled)
            self.mg.add_edge(n1, n2, weight=0.9)
            dfs_nodes = self.mg.dfs_recall(n1, weight_threshold=0.5)
            self.assertIn(n1, dfs_nodes)
            self.assertIn(n2, dfs_nodes)

        def test_recency_decay(self):
            n1 = self.mg.add_node("Old memory")
            self.mg.graph.nodes[n1]["timestamp"] -= 3600  # 1 hour ago
            old_recency = self.mg.graph.nodes[n1]["recency"]
            self.mg.recency_decay(decay_factor=0.9)
            new_recency = self.mg.graph.nodes[n1]["recency"]
            self.assertLess(new_recency, old_recency)

        def test_consolidation(self):
            for i in range(5):
                self.mg.add_node(f"Content {i}")
            pruned = self.mg.consolidate(k=2)
            self.assertIsInstance(pruned, list)

    class TestRLAgent(unittest.TestCase):
        def setUp(self):
            self.mg = MemoryGraph("RLGraph")
            self.agent = RLAgent(learning_rate=0.1, discount_factor=0.9)
            # Create a simple chain.
            self.n1 = self.mg.add_node("Start")
            self.n2 = self.mg.add_node("Middle")
            self.n3 = self.mg.add_node("End")
            self.mg.add_edge(self.n1, self.n2, weight=0.5)
            self.mg.add_edge(self.n2, self.n3, weight=0.5)

        def test_q_update(self):
            q_initial = self.agent.get_q(self.n1, self.n2)
            self.agent.update_q(self.n1, self.n2, reward=1.0)
            q_new = self.agent.get_q(self.n1, self.n2)
            self.assertGreater(q_new, q_initial)

        def test_apply_reward(self):
            path = [self.n1, self.n2, self.n3]
            self.agent.apply_reward(self.mg, path, reward=1.0)
            # Check that edge weights are updated to reflect the new Q-values.
            self.assertAlmostEqual(self.mg.graph[self.n1][self.n2]["weight"], self.agent.get_q(self.n1, self.n2))
            self.assertAlmostEqual(self.mg.graph[self.n2][self.n3]["weight"], self.agent.get_q(self.n2, self.n3))

    class TestMAGSSystem(unittest.TestCase):
        def setUp(self):
            # Dummy LLM that echoes the prompt.
            class DummyLLM:
                def query(self, prompt: str) -> str:
                    return f"LLM Response: {prompt}"
            self.dummy_llm = DummyLLM()
            self.ep_mem = EpisodicMemoryBlock()
            self.sem_mem = SemanticMemoryBlock()
            self.agent = RLAgent(learning_rate=0.1, discount_factor=0.9)
            self.mags = MAGS(self.dummy_llm, self.ep_mem, self.sem_mem, self.agent)

        def test_process_input(self):
            prompt = "Tell me about neural networks."
            response = self.mags.process_input(prompt)
            self.assertTrue(response.startswith("LLM Response:"))
            self.assertGreater(len(self.ep_mem.get_all_nodes()), 0)
            self.assertGreater(len(self.sem_mem.get_all_nodes()), 0)

        def test_consolidate_memory(self):
            for text in ["fact one", "fact two", "fact three"]:
                self.ep_mem.add_node(text)
            pruned = self.mags.consolidate_memory()
            self.assertIsInstance(pruned, list)

    class TestGameSimulator(unittest.TestCase):
        def setUp(self):
            class DummyLLM:
                def query(self, prompt: str) -> str:
                    return f"Response to: {prompt}"
            self.dummy_llm = DummyLLM()
            self.ep_mem = EpisodicMemoryBlock()
            self.sem_mem = SemanticMemoryBlock()
            self.agent = RLAgent(learning_rate=0.1, discount_factor=0.9)
            self.mags = MAGS(self.dummy_llm, self.ep_mem, self.sem_mem, self.agent)
            self.game = GameSimulator(self.mags)

        def test_simulation(self):
            results, pruned = self.game.run_simulation(rounds=4)
            self.assertEqual(len(results), 4)
            self.assertIsInstance(pruned, list)

    # Run all tests.
    unittest.main(verbosity=2)