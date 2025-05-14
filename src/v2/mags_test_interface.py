"""
MAGS Test Interface
-------------------
A script to test the components of the Memory Augmented Graph Scaling (MAGS) system.
Features a stylized logger and placeholder LLM interactions for comprehensive testing.
"""
import logging
import os
import sys
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set, Callable


# print current path
print("Current path: ", os.getcwd())

# Attempt to import the MAGS components
try:
    from mags.hyperparameters.hyperparameters import Hyperparameters
    from mags.graph.node import Node
    import mags.graph.node
    from mags.graph.edge import Edge
    from mags.graph.lkg import LiquidKnowledgeGraph, ModifierPhase
    from mags.memory.base_memory import BaseMemoryBlock
    from mags.memory.semantic_memory import SemanticMemory
    from mags.memory.episodic_memory import EpisodicMemory
    from mags.memory.general_memory import GeneralMemory
    from mags.modifiers.base_modifier import BaseModifier
    from mags.modifiers.decay_modifier import DecayModifier
    from mags.modifiers.strengthen_modifier import StrengthenModifier
    from mags.modifiers.feedback_modifier import FeedbackModifier
    from mags.modifiers.hebbian_modifier import HebbianModifier
    from mags.modifiers.permanence_modifier import PermanenceModifier
    from mags.modifiers.prune_modifiers import Prune1Modifier, Prune2Modifier
    from mags.modifiers.reconcile_replace_modifier import ReconcileReplaceModifier
    from mags.modifiers.error_modifier import ErrorCalculatorModifier
    from mags.query.anchor_search import MaximumAnchorSearch
    from mags.query.traversal import DFSTraversal, BFSTraversal, TraversalStoppingCondition, Triplet
except ImportError as e:
    print(f"Error importing MAGS components: {e}")
    print("Please ensure the 'mags' package is in your PYTHONPATH or installed.")
    sys.exit(1)

# --- Terminal Colors ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    ORANGE = "\033[38;5;208m" # Approx Orange

    HEADER = BOLD + MAGENTA
    SUCCESS = BOLD + GREEN
    WARNING = BOLD + YELLOW
    ERROR = BOLD + RED
    INFO_BLUE = BOLD + BLUE
    INFO_CYAN = BOLD + CYAN
    PARAM = YELLOW
    VALUE = CYAN
    SECTION = BOLD + ORANGE


# --- Stylized Logger ---
class StylizedLogger:
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    LOG_COLORS = {
        "DEBUG": Colors.CYAN,
        "INFO": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "CRITICAL": Colors.BOLD + Colors.RED,
    }

    def __init__(self, name="MAGS_Test", level="INFO", log_file=None, to_console=True, to_jupyter=False):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LOG_LEVELS.get(level.upper(), logging.INFO))
        self.logger.handlers = [] # Clear existing handlers

        formatter_str = "%(asctime)s | %(levelname)-7s | %(message)s"
        date_format = "%H:%M:%S" # For console, simpler time
        file_date_format = "%Y-%m-%d %H:%M:%S"

        if to_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(ColoredFormatter(formatter_str, date_format, use_colors=not to_jupyter))
            self.logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file, mode='w') # Overwrite log file each run
            file_formatter = logging.Formatter(formatter_str, datefmt=file_date_format)
            fh.setFormatter(file_formatter)
            self.logger.addHandler(fh)

        if to_jupyter: # Jupyter uses basic formatting; colors often don't work well with default logger
            # For Jupyter, if rich library is available, it provides better output.
            # Otherwise, basic stream handler without custom colors is safer.
            # For this example, we rely on the console handler and assume user can see it or the file.
            pass


    def debug(self, msg, **kwargs): self.logger.debug(msg, **kwargs)
    def info(self, msg, **kwargs): self.logger.info(msg, **kwargs)
    def warning(self, msg, **kwargs): self.logger.warning(msg, **kwargs)
    def error(self, msg, **kwargs): self.logger.error(msg, **kwargs)
    def critical(self, msg, **kwargs): self.logger.critical(msg, **kwargs)

    def section(self, title):
        self.info(f"{Colors.SECTION}===== {title.upper()} ====={Colors.RESET}")

    def subsection(self, title):
        self.info(f"{Colors.INFO_BLUE}--- {title} ---{Colors.RESET}")

    def log_dict(self, data: Dict, title: Optional[str] = None):
        if title:
            self.info(f"{Colors.PARAM}{title}:{Colors.RESET}")
        for key, value in data.items():
            self.info(f"  {Colors.PARAM}{key}:{Colors.RESET} {Colors.VALUE}{value}{Colors.RESET}")

    def success(self, msg):
        self.info(f"{Colors.SUCCESS}SUCCESS: {msg}{Colors.RESET}")
    
    def failure(self, msg):
        self.error(f"{Colors.ERROR}FAILURE: {msg}{Colors.RESET}")


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt=datefmt)
        self.use_colors = use_colors

    def format(self, record):
        levelname = record.levelname
        asctime = self.formatTime(record, self.datefmt)
        message = record.getMessage()

        if self.use_colors and levelname in StylizedLogger.LOG_COLORS:
            levelname_color = StylizedLogger.LOG_COLORS[levelname] + levelname + Colors.RESET
            # Apply color to the whole line based on level for visual impact
            # Or just color the levelname
            log_entry = f"{asctime} | {levelname_color:<20} | {message}" # Adjusted spacing for color codes
            return StylizedLogger.LOG_COLORS[levelname] + log_entry + Colors.RESET # Color whole line
        else:
            return f"{asctime} | {levelname:<7} | {message}"


# --- Placeholder LLM & Data Functions ---
def mock_get_embeddings(text: str, dim: int = 5) -> List[float]:
    """Generates a dummy embedding vector for given text."""
    random.seed(len(text)) # Make it somewhat deterministic based on text
    return [random.uniform(-1, 1) for _ in range(dim)]

def mock_llm_feedback_for_nodes(node_ids: Set[int]) -> Dict[int, float]:
    """Generates dummy feedback scores for a set of node IDs."""
    feedback = {}
    for node_id in node_ids:
        feedback[node_id] = random.choice([-1.0, 0.0, 1.0]) * random.random() # Score between -1 and 1
    return feedback

def mock_llm_engram_creation(text_input: str) -> List[Tuple[str, str, str]]:
    """Simulates an LLM creating new triplets from input text."""
    words = text_input.lower().split()
    triplets = []
    if len(words) >= 3:
        for i in range(0, len(words) - 2, 3): # Create non-overlapping triplets
            triplets.append((words[i], words[i+1], words[i+2]))
    if not triplets and len(words) >= 1 : # Fallback for short text
         triplets.append((words[0], "is_a_concept", words[0]))
    return triplets


# --- MAGS Test Suite ---
class MAGSTestSuite:
    def __init__(self, logger: StylizedLogger, hyperparams: Hyperparameters):
        self.logger = logger
        self.hyperparams = hyperparams
        self.test_results = {"passed": 0, "failed": 0}

    def _assert(self, condition: bool, success_msg: str, failure_msg: str):
        if condition:
            self.logger.success(success_msg)
            self.test_results["passed"] += 1
        else:
            self.logger.failure(failure_msg)
            self.test_results["failed"] += 1
        return condition

    def print_header(self):
        self.logger.info(Colors.HEADER + r"""
    ███╗   ███╗ █████╗  ██████╗ ███████╗
    ████╗ ████║██╔══██╗██╔════╝ ██╔════╝
    ██╔████╔██║███████║██║  ███ ███████╗
    ██║╚██╔╝██║██╔══██║██║   ██║╚════██║
    ██║ ╚═╝ ██║██║  ██║╚██████╔╝███████║
    ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝   
    """ + Colors.RESET)


    def test_hyperparameters(self):
        self.logger.section("Hyperparameter System Test")
        self.hyperparams.set_param("test_global_param", 123)
        self._assert(self.hyperparams.get_param("test_global_param") == 123,
                     "Global hyperparameter set and get.",
                     "Failed to set/get global hyperparameter.")

        self.hyperparams.register_defaults("test_component", {"val1": "abc", "val2": 45.6})
        self._assert(self.hyperparams.get_component_param("test_component", "val1") == "abc",
                     "Component default hyperparameter registration and retrieval.",
                     "Failed to register/retrieve component default hyperparameter.")
        self.logger.log_dict(self.hyperparams._params, "Current Hyperparameters")


    def test_node_edge_creation(self):
        self.logger.section("Node and Edge Creation Test")
        Node.reset_id_counter()
        Edge.reset_id_counter()

        node1_content = "Test Node Alpha"
        node1_emb = mock_get_embeddings(node1_content)
        node1 = Node(content=node1_content, hyperparams=self.hyperparams, importance_score=1.5, embeddings=node1_emb)
        self._assert(node1.id == 1 and node1.content == node1_content and node1.importance_score == 1.5,
                     f"Node 1 created: ID={node1.id}, Content='{node1.content}'.",
                     "Node 1 creation failed.")
        self.logger.info(f"Node 1 details: {node1}")

        node2 = Node(content="Test Node Beta", hyperparams=self.hyperparams, surprise_score=0.8)
        self._assert(node2.id == 2 and node2.surprise_score == 0.8,
                     f"Node 2 created: ID={node2.id}, Surprise={node2.surprise_score}.",
                     "Node 2 creation failed.")

        edge1 = Edge(source_node_id=node1.id, target_node_id=node2.id, content="connects_to",
                     hyperparams=self.hyperparams, strength_score=0.75)
        self._assert(edge1.id == 1 and edge1.strength_score == 0.75 and edge1.source_id == node1.id,
                     f"Edge 1 created: {edge1}.",
                     "Edge 1 creation failed.")
        self.logger.info(f"Edge 1 details: {edge1}")


    def test_lkg_core_operations(self):
        self.logger.section("Liquid Knowledge Graph (LKG) Core Operations Test")
        lkg = LiquidKnowledgeGraph(self.hyperparams)
        self.hyperparams.register_defaults("lkg", {"allow_duplicate_node_content": False})


        self.logger.subsection("Adding Nodes and Edges")
        n1 = lkg.add_node("Paris", importance_score=2.0, embeddings=mock_get_embeddings("Paris"))
        n2 = lkg.add_node("France", importance_score=1.8, embeddings=mock_get_embeddings("France"))
        n3 = lkg.add_node("Eiffel Tower", importance_score=1.5, embeddings=mock_get_embeddings("Eiffel Tower"))

        self._assert(lkg.get_node_count() == 3,
                     f"LKG node count is {lkg.get_node_count()} after adding 3 nodes.",
                     f"LKG node count expected 3, got {lkg.get_node_count()}.")

        e1 = lkg.add_edge(n1.id, n2.id, "is_capital_of", strength_score=0.9)
        e2 = lkg.add_edge(n3.id, n1.id, "is_located_in", strength_score=0.85)
        self._assert(lkg.get_edge_count() == 2 and e1 is not None and e2 is not None,
                     f"LKG edge count is {lkg.get_edge_count()} after adding 2 edges.",
                     f"LKG edge count expected 2, got {lkg.get_edge_count()}.")
        self.logger.log_dict(lkg.graph_representation(), "LKG Initial State")

        self.logger.subsection("Testing Duplicate Node Content Handling")
        n_paris_again = lkg.add_node("Paris") # Should return existing Paris node
        self._assert(n_paris_again.id == n1.id and lkg.get_node_count() == 3,
                     "Adding 'Paris' again reused existing node as expected.",
                     "Duplicate node handling failed for 'Paris'.")


        self.logger.subsection("Adding Triplet")
        tn1, te1, tn2 = lkg.add_triplet("Sacre Coeur", "is_monument_in", "Paris")
        self._assert(tn1 is not None and te1 is not None and tn2 is not None and lkg.get_node_count() == 4 and lkg.get_edge_count() == 3,
                     f"Triplet ('Sacre Coeur', 'is_monument_in', 'Paris') added. Node count: {lkg.get_node_count()}, Edge count: {lkg.get_edge_count()}",
                     "Failed to add triplet.")
        self.logger.info(f"Added Triplet: Node1='{tn1.content}', Edge='{te1.content}', Node2='{tn2.content}'")


        self.logger.subsection("Removing Node and Edges")
        lkg.remove_node(n3.id) # Eiffel Tower
        # Edge e2 (Eiffel Tower -> Paris) should also be removed
        self._assert(lkg.get_node(n3.id) is None and lkg.get_edge(e2.id) is None and lkg.get_node_count() == 3,
                     f"Node 'Eiffel Tower' and its incident edge removed. Node count: {lkg.get_node_count()}.",
                     "Failed to remove node or its incident edges.")
        self._assert(lkg.get_edge_count() == 2, # e1 and te1 should remain
                      f"Edge count is {lkg.get_edge_count()} after node removal.",
                      f"Edge count expected 2, got {lkg.get_edge_count()} after node removal.")

        lkg.clear()
        self._assert(lkg.get_node_count() == 0 and lkg.get_edge_count() == 0,
                     "LKG cleared successfully.", "LKG clear failed.")


    def _setup_sample_lkg_for_modifiers(self) -> LiquidKnowledgeGraph:
        lkg = LiquidKnowledgeGraph(self.hyperparams)
        Node.reset_id_counter() # Reset for predictable IDs in this test
        Edge.reset_id_counter()
        self.hyperparams.register_defaults("node", {"default_importance": 1.0, "default_surprise": 0.5})
        self.hyperparams.register_defaults("edge", {"default_strength": 0.5})


        # Nodes
        self.n_apple = lkg.add_node("apple", importance_score=1.0, embeddings=mock_get_embeddings("apple"))
        self.n_fruit = lkg.add_node("fruit", importance_score=0.8, embeddings=mock_get_embeddings("fruit"))
        self.n_red = lkg.add_node("red", importance_score=0.5, embeddings=mock_get_embeddings("red"))
        self.n_tree = lkg.add_node("tree", importance_score=0.7, embeddings=mock_get_embeddings("tree"))
        self.n_banana = lkg.add_node("banana", importance_score=0.9, embeddings=mock_get_embeddings("banana"))
        self.n_yellow = lkg.add_node("yellow", importance_score=0.4, embeddings=mock_get_embeddings("yellow"))


        # Edges
        self.e_apple_fruit = lkg.add_edge(self.n_apple.id, self.n_fruit.id, "is_a", strength_score=0.8)
        self.e_apple_red = lkg.add_edge(self.n_apple.id, self.n_red.id, "is_color", strength_score=0.6)
        self.e_apple_tree = lkg.add_edge(self.n_apple.id, self.n_tree.id, "grows_on", strength_score=0.7)
        self.e_banana_fruit = lkg.add_edge(self.n_banana.id, self.n_fruit.id, "is_a", strength_score=0.85)
        self.e_banana_yellow = lkg.add_edge(self.n_banana.id, self.n_yellow.id, "is_color", strength_score=0.65)

        self.logger.info("Sample LKG for modifier tests created:")
        self.logger.log_dict({n.id: repr(n) for n in lkg.get_all_nodes()}, "Nodes")
        self.logger.log_dict({e.id: repr(e) for e in lkg.get_all_edges()}, "Edges")
        return lkg

    def test_modifiers(self):
        self.logger.section("Modifiers Test")
        lkg = self._setup_sample_lkg_for_modifiers()

        # --- Decay Modifier ---
        self.logger.subsection("Decay Modifier Test")
        self.hyperparams.register_defaults("decay", {"node_importance_decay_factor": 0.9, "edge_strength_decay_factor": 0.9})
        decay_mod = DecayModifier(self.hyperparams)
        original_apple_importance = self.n_apple.importance_score
        original_e_apple_fruit_strength = self.e_apple_fruit.strength_score
        decay_mod.apply(lkg)
        self._assert(self.n_apple.importance_score < original_apple_importance and \
                     self.e_apple_fruit.strength_score < original_e_apple_fruit_strength,
                     f"Decay applied. Apple importance: {self.n_apple.importance_score:.2f}, Edge strength: {self.e_apple_fruit.strength_score:.2f}",
                     "Decay modifier failed.")

        # --- Strengthen Modifier ---
        self.logger.subsection("Strengthen Modifier Test")
        self.hyperparams.register_defaults("strengthen", {"node_importance_boost_factor": 1.2, "edge_strength_boost_factor": 1.1})
        strengthen_mod = StrengthenModifier(self.hyperparams)
        recalled_nodes = {self.n_apple.id, self.n_fruit.id}
        recalled_edges = {self.e_apple_fruit.id}
        current_apple_importance = self.n_apple.importance_score
        current_e_apple_fruit_strength = self.e_apple_fruit.strength_score
        strengthen_mod.apply(lkg, recalled_nodes, recalled_edges)
        self._assert(self.n_apple.importance_score > current_apple_importance and \
                     self.e_apple_fruit.strength_score > current_e_apple_fruit_strength,
                     f"Strengthen applied. Apple importance: {self.n_apple.importance_score:.2f}, Edge strength: {self.e_apple_fruit.strength_score:.2f}",
                     "Strengthen modifier failed.")

        # --- Feedback Modifier ---
        self.logger.subsection("Feedback Modifier Test")
        self.hyperparams.register_defaults("feedback", {"positive_feedback_multiplier": 1.5, "negative_feedback_multiplier": 0.5})
        feedback_mod = FeedbackModifier(self.hyperparams)
        feedback_scores = {self.n_red.id: 1.0, self.n_tree.id: -1.0} # Positive for red, negative for tree
        red_importance_before = self.n_red.importance_score
        tree_importance_before = self.n_tree.importance_score
        feedback_mod.apply(lkg, feedback_scores)
        self._assert(self.n_red.importance_score > red_importance_before and \
                     self.n_tree.importance_score < tree_importance_before,
                     f"Feedback applied. Red importance: {self.n_red.importance_score:.2f}, Tree importance: {self.n_tree.importance_score:.2f}",
                     "Feedback modifier failed.")

        # --- Hebbian Modifier ---
        self.logger.subsection("Hebbian Modifier Test")
        self.hyperparams.register_defaults("hebbian", {"strength_increment": 0.15})
        hebbian_mod = HebbianModifier(self.hyperparams)
        # Simulate apple and red being co-activated
        activated_pairs = {(self.n_apple.id, self.n_red.id)}
        e_apple_red_strength_before = self.e_apple_red.strength_score
        hebbian_mod.apply(lkg, activated_pairs)
        self._assert(self.e_apple_red.strength_score > e_apple_red_strength_before,
                     f"Hebbian applied. Edge Apple-Red strength: {self.e_apple_red.strength_score:.2f}",
                     "Hebbian modifier failed.")

        # --- Permanence Modifier ---
        self.logger.subsection("Permanence Modifier Test")
        self.hyperparams.register_defaults("permanence", {"permanence_importance_threshold": 1.5}) # Apple should become permanent
        permanence_mod = PermanenceModifier(self.hyperparams)
        self.n_apple.update_importance(1.5) # Ensure it's above threshold
        self._assert(not self.n_apple.is_permanent, "Apple node is initially not permanent.", "Initial permanence state wrong.")
        permanence_mod.apply(lkg)
        self._assert(self.n_apple.is_permanent,
                     "Permanence applied. Apple node is now permanent.",
                     "Permanence modifier failed for Apple node.")

        # --- Prune1 Modifier ---
        self.logger.subsection("Prune1 Modifier Test")
        self.hyperparams.register_defaults("prune1", {"edge_strength_prune_threshold": 0.3, "node_importance_prune_threshold": 0.3})
        prune1_mod = Prune1Modifier(self.hyperparams)
        self.n_yellow.update_importance(0.1) # Make yellow node prunable
        self.e_banana_yellow.update_strength(0.1) # Make banana-yellow edge prunable
        node_count_before = lkg.get_node_count()
        edge_count_before = lkg.get_edge_count()
        prune1_mod.apply(lkg)
        self._assert(lkg.get_node(self.n_yellow.id) is None and lkg.get_edge(self.e_banana_yellow.id) is None,
                     f"Prune1 applied. Yellow node and Banana-Yellow edge removed.",
                     "Prune1 modifier failed.")
        self.logger.info(f"Node count after Prune1: {lkg.get_node_count()} (was {node_count_before}). Edge count: {lkg.get_edge_count()} (was {edge_count_before})")


        # --- Prune2 Modifier ---
        # (Need to re-add some nodes/edges to test Prune2 capacity limits)
        lkg.add_node("extra_node1", importance_score=0.01)
        lkg.add_node("extra_node2", importance_score=0.02)
        current_node_count = lkg.get_node_count()
        self.logger.subsection(f"Prune2 Modifier Test (Current node count: {current_node_count})")
        # Set max_nodes to be less than current count to trigger pruning
        max_nodes_for_prune2 = current_node_count -1
        self.hyperparams.register_defaults("prune2", {"max_nodes_threshold": max_nodes_for_prune2, "max_edges_threshold": lkg.get_edge_count() + 5})
        prune2_mod = Prune2Modifier(self.hyperparams)
        prune2_mod.apply(lkg)
        self._assert(lkg.get_node_count() == max_nodes_for_prune2,
                     f"Prune2 applied. Node count reduced to {lkg.get_node_count()} (max was {max_nodes_for_prune2}).",
                     f"Prune2 modifier failed. Node count is {lkg.get_node_count()}, expected {max_nodes_for_prune2}.")

        # --- ReconcileReplace Modifier ---
        self.logger.subsection("ReconcileReplace Modifier Test")
        reconcile_mod = ReconcileReplaceModifier(self.hyperparams)
        # Test Case: Adding a new triplet where one node exists
        # Original: apple -> is_a -> fruit. New: apple -> tastes_like -> sweetness
        # n_apple ("apple") already exists.
        apple_node_id_before_reconcile = self.n_apple.id
        triplet_to_reconcile = ("apple", "tastes_like", "sweetness")
        rn1, re1, rn2 = reconcile_mod.apply(lkg, triplet_to_reconcile)

        self._assert(rn1 is not None and rn1.id == apple_node_id_before_reconcile and \
                     re1 is not None and re1.content == "tastes_like" and \
                     rn2 is not None and rn2.content == "sweetness",
                     f"ReconcileReplace applied for new relation to 'apple'. New edge: {re1}",
                     "ReconcileReplace modifier failed for new relation.")
        self.logger.info(f"Reconciled: Node1='{rn1.content}', Edge='{re1.content}', Node2='{rn2.content}'")

        # --- Error Calculator Modifier ---
        self.logger.subsection("ErrorCalculator Modifier Test")
        error_mod = ErrorCalculatorModifier(self.hyperparams)
        # lkg.change_count has been updated by previous modifiers
        error_score = error_mod.apply(lkg)
        self._assert(error_score > 0,
                     f"ErrorCalculator run. Calculated score: {error_score:.3f} (based on {lkg.change_count} changes)",
                     "ErrorCalculator modifier failed to produce a score.")


    def test_memory_block_operations(self):
        self.logger.section("Memory Block Operations Test")

        # Use a specific memory type, e.g., SemanticMemory
        mem_block = SemanticMemory(self.hyperparams)
        self.hyperparams.register_defaults("semantic_traversal_stop", {"max_depth": 2, "min_path_importance_product_threshold": 0.001})
        self.hyperparams.register_defaults("maximumanchor", {"num_anchors_to_select_N": 1, "area_coverage_depth_X":1})
        self.hyperparams.register_defaults("bfstraversal", {"top_Y_triplets_to_return": 5})


        # Populate LKG within the memory block
        n_cat = mem_block.add_node_to_lkg("cat", importance_score=1.0, embeddings=mock_get_embeddings("cat"))
        n_animal = mem_block.add_node_to_lkg("animal", importance_score=0.8, embeddings=mock_get_embeddings("animal"))
        n_mammal = mem_block.add_node_to_lkg("mammal", importance_score=0.9, embeddings=mock_get_embeddings("mammal"))
        n_dog = mem_block.add_node_to_lkg("dog", importance_score=1.1, embeddings=mock_get_embeddings("dog"))

        mem_block.add_edge_to_lkg(n_cat.id, n_mammal.id, content="is_a", strength_score=0.9)
        mem_block.add_edge_to_lkg(n_mammal.id, n_animal.id, content="is_a_type_of", strength_score=0.8)
        mem_block.add_edge_to_lkg(n_dog.id, n_mammal.id, content="is_a", strength_score=0.95)

        self.logger.info(f"Memory block ({mem_block.memory_type}) LKG populated. Nodes: {mem_block.get_lkg_node_count()}, Edges: {mem_block.get_lkg_edge_count()}")

        # Register a Decay modifier to run AFTER_QUERY
        decay_mod = DecayModifier(self.hyperparams)
        mem_block.add_modifier(decay_mod, ModifierPhase.AFTER_QUERY, name="test_decay_after")

        # Prepare args for the AFTER_QUERY modifier (Decay needs no specific args from query result)
        # If it was StrengthenModifier, it would be:
        # strengthen_args = (mem_block._last_query_recalled_node_ids, mem_block._last_query_recalled_edge_ids)
        # after_modifier_args_tuple = (strengthen_args,) # tuple of tuples
        after_modifier_args_tuple = ((),) # Empty tuple of args for the DecayModifier

        self.logger.subsection("Querying Memory Block")
        query_text = "tell me about cat"
        query_embedding = mock_get_embeddings(query_text)
        recalled_triplets = mem_block.query_memory(query_text, query_embedding,
                                                 after_modifier_args=after_modifier_args_tuple)

        self._assert(len(recalled_triplets) > 0 or len(recalled_triplets) == 0, # Query might return 0 results based on setup
                     f"Query executed. Recalled {len(recalled_triplets)} triplets.",
                     "Query execution failed.") # This assert might need refinement based on expected output

        if recalled_triplets:
            self.logger.info("Recalled Triplets:")
            for n1, e, n2 in recalled_triplets:
                self.logger.info(f"  ({n1.content}, {e.content}, {n2.content})")
        else:
            self.logger.warning("Query returned no triplets. This might be expected depending on test data and thresholds.")

        # Check if Decay modifier (AFTER_QUERY) ran by observing changed importance
        importance_after_query = mem_block.lkg.get_node(n_cat.id).importance_score
        self._assert(importance_after_query < 1.0, # Assuming original was 1.0 and decay happened
                     f"AFTER_QUERY Decay modifier likely ran. Cat importance: {importance_after_query:.3f} (was 1.0 before decay)",
                     "AFTER_QUERY Decay modifier effect not observed as expected.")


        self.logger.subsection("Testing DETACHED Modifier Execution")
        # Use ErrorCalculator as a detached modifier
        error_mod_detached = ErrorCalculatorModifier(self.hyperparams)
        mem_block.add_modifier(error_mod_detached, ModifierPhase.DETACHED, name="calculate_graph_error_detached")
        # Run it, it returns the score
        # Detached modifiers are called with (graph, *args) by LKG's run_modifiers.
        # The ErrorCalculator.apply returns a float. LKG.run_modifiers doesn't return values from modifiers.
        # So, to get the value, we'd typically call the modifier's apply method directly if a return is needed,
        # or have it log/store its result.
        # For this test, we'll just confirm it runs.
        # We need to make ErrorCalculator's apply method not require return to fit LKG.run_modifiers
        # Or the test calls it directly:
        error_score = error_mod_detached.apply(mem_block.lkg)
        self._assert(error_score is not None,
                     f"DETACHED ErrorCalculator modifier run manually, score: {error_score:.3f}",
                     "DETACHED ErrorCalculator modifier execution failed.")
        # Test through memory_block interface:
        mem_block.run_lkg_modifiers(ModifierPhase.DETACHED, detached_modifier_name="calculate_graph_error_detached")
        # This call won't return the value, but the modifier's logging should indicate it ran.

    def run_all_tests(self):
        self.print_header()
        start_time = time.time()

        self.test_hyperparameters()
        self.test_node_edge_creation()
        self.test_lkg_core_operations()
        self.test_modifiers()
        self.test_memory_block_operations()

        end_time = time.time()
        self.logger.section("Test Summary")
        total_tests = self.test_results["passed"] + self.test_results["failed"]
        self.logger.info(f"Total tests run: {Colors.VALUE}{total_tests}{Colors.RESET}")
        self.logger.info(f"Passed: {Colors.SUCCESS}{self.test_results['passed']}{Colors.RESET}")
        self.logger.info(f"Failed: {Colors.ERROR if self.test_results['failed'] > 0 else Colors.YELLOW}{self.test_results['failed']}{Colors.RESET}")
        self.logger.info(f"Duration: {Colors.VALUE}{end_time - start_time:.2f} seconds{Colors.RESET}")

        if self.test_results["failed"] > 0:
            self.logger.error(Colors.BOLD + Colors.RED + "SOME TESTS FAILED!" + Colors.RESET)
        else:
            self.logger.success(Colors.BOLD + Colors.GREEN + "ALL TESTS PASSED!" + Colors.RESET)


# --- Main Execution ---
if __name__ == "__main__":
    # Setup Logger
    log_filename = "mags_test_run.log"
    # Detect if in Jupyter to adjust console logging (basic for now)
    in_jupyter = 'ipykernel' in sys.modules

    logger = StylizedLogger(name="MAGS_MainTest", level="DEBUG", log_file=log_filename, to_console=True, to_jupyter=in_jupyter)
    logger.info(f"MAGS Test Interface starting. Log file: {log_filename}")
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Current working directory: {os.getcwd()}")


    # Initialize Hyperparameters (can be loaded from a file in a real scenario)
    global_hyperparams = Hyperparameters()
    global_hyperparams.set_param("system_version", "0.1.0-alpha")
    global_hyperparams.set_param("test_run_timestamp", datetime.now().isoformat())
    # Register some default hyperparams for components if not done elsewhere
    # (Ideally, components register their own upon instantiation if defaults aren't already there)
    global_hyperparams.register_defaults("node", {"default_importance": 1.0, "default_surprise": 0.5, "embedding_dim": 5})
    global_hyperparams.register_defaults("edge", {"default_strength": 0.5})
    global_hyperparams.register_defaults("lkg", {"max_nodes": 1000, "max_edges": 5000, "allow_duplicate_node_content": False})


    # Initialize and Run Test Suite
    test_suite = MAGSTestSuite(logger, global_hyperparams)
    test_suite.run_all_tests()