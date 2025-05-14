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
    from mags.inference.generic_llm import GnicLLMWrapper
    from mags.inference.Embedder import LocalEmbedder
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


class MAGSGenPipeline:
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


    def log_params(self):
        self.logger.log_dict(self.hyperparams._params, "Current Hyperparameters")






    def _setup_lkg(self) -> LiquidKnowledgeGraph:

        self.hyperparams.register_defaults("node", {"default_importance": 1.0, "default_surprise": 0.5})
        self.hyperparams.register_defaults("edge", {"default_strength": 1.0})
        self.hyperparams.register_defaults("semantic_traversal_stop", {"max_depth": 2, "min_path_importance_product_threshold": 0.1})
        self.hyperparams.register_defaults("maximumanchor", {"num_anchors_to_select_N": 5, "area_coverage_depth_X":3})
        self.hyperparams.register_defaults("bfstraversal", {"top_Y_triplets_to_return": 7})
        decay_mod = DecayModifier(self.hyperparams)

        mem_block = SemanticMemory(self.hyperparams)
        Node.reset_id_counter() 
        Edge.reset_id_counter()

        mem_block.add_modifier(decay_mod, ModifierPhase.AFTER_QUERY, name="test_decay_after")
        after_modifier_args_tuple = ((),)

        reconcile_mod = ReconcileReplaceModifier(self.hyperparams)
        mem_block.add_modifier(reconcile_mod, ModifierPhase.DETACHED, name="reconcile")


        return mem_block

    def run(self, lkg, query):
        
        self.agent = GnicLLMWrapper(api_key="", model="gpt-4o-mini")

        self.embedder = LocalEmbedder()

        ROUTE_PROMPT = """You have the following query. Please determine if it is a statement or a question. If it includes at LEAST one question 
        output "Q". If it includes ONLY statement(s) or unknow, output "N". Output ONLY Q or N, NOTHING ELSE. Your response will be parsed
        by a downstream agent."""

        route_result = self.agent._call_openai(messages = [
            {
                "role": "system",
                "content": ROUTE_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ])["choices"][0]["message"]["content"]

        new_triplet_q = query
        
        if route_result == "Q":
            logger.info(Colors.INFO_CYAN + "Query Pipeline Reached" + Colors.RESET)
            questions = self.agent.extract_asks(query)
            
            logger.info(Colors.INFO_CYAN + "Developed " + Colors.GREEN + Colors.UNDERLINE + str(len(questions)) + Colors.INFO_CYAN + " Recalls" + Colors.RESET)

            conjoin = "\n\n".join(questions)

            embedding = self.embedder.embed_text(conjoin)

            recalled_triplets = lkg.query_memory(conjoin, embedding)

            logger.info(Colors.INFO_CYAN + "Recalled " + Colors.GREEN + Colors.UNDERLINE + str(len(recalled_triplets)) + Colors.INFO_CYAN + " triplets" + Colors.RESET)

            normalize_triplets = []

            for n1, e, n2 in recalled_triplets:
                normalize_triplets.append([n1.content, e.content, n2.content])
            
            answer = self.agent.answer_question(query, normalize_triplets)
            
            logger.info(Colors.INFO_CYAN + "Answer: " + Colors.GREEN + Colors.UNDERLINE + answer + Colors.INFO_CYAN + Colors.RESET)
        

        elif route_result == "N":
            logger.info(Colors.INFO_CYAN + "Engram Pipeline Reached" + Colors.RESET)
        else:
            logger.failure("UNKNOWN RESPONSE FROM ROUTE AGENT")
            return

        new_triplets = self.agent.extract_triplets(new_triplet_q)
        logger.info(Colors.INFO_CYAN + "Found " + Colors.GREEN + Colors.UNDERLINE + str(len(new_triplets)) + Colors.INFO_CYAN + " New Candidates" + Colors.RESET)

        existing_triplets_unstruct = lkg.get_all_triplets()
        reconcilled = self.agent.reconcile_triplets(new_triplets=new_triplets, existing_triplets=existing_triplets_unstruct)
        logger.info(Colors.INFO_CYAN + "Found " + Colors.GREEN + Colors.UNDERLINE + str(len(reconcilled)) + Colors.INFO_CYAN + " Adjustments" + Colors.RESET)

        reconcile_mod = ReconcileReplaceModifier(self.hyperparams)
        triplet_to_reconcile = ("apple", "tastes_like", "sweetness")
        rn1, re1, rn2 = reconcile_mod.apply(lkg.lkg, triplet_to_reconcile)

        add = 0
        recon = 0

        for rec in reconcilled:
            if(rec["id"] == "new"):
                lkg.add_triplet_to_lkg(rec["triplet"][0], rec["triplet"][1], rec["triplet"][2])
                add += 1
            else:
                triplet = ((rec["triplet"][0], rec["triplet"][1], rec["triplet"][2]),)
                lkg.run_lkg_modifiers(ModifierPhase.DETACHED, triplet, detached_modifier_name="reconcile")
                recon += 1
        

        logger.info(Colors.INFO_CYAN + "Added " + Colors.GREEN + Colors.UNDERLINE + str((add)) + Colors.INFO_CYAN + " Triplets" + Colors.RESET)
        logger.info(Colors.INFO_CYAN + "Reconcilied " + Colors.GREEN + Colors.UNDERLINE + str((recon)) + Colors.INFO_CYAN + " Triplets" + Colors.RESET)

        num_nodes = lkg.get_lkg_node_count()
        num_edges = lkg.get_lkg_edge_count()

        logger.info(Colors.INFO_CYAN + "MAGS has " + Colors.GREEN + Colors.UNDERLINE + str((num_nodes)) + " Nodes, " + str((num_edges)) + " Edges, " + Colors.RESET)

    def loop(self):
        self.print_header()
        start_time = time.time()
        lkg = self._setup_lkg()
        self.log_params()

        logger.info(Colors.INFO_CYAN + "MAGS is " + Colors.GREEN + Colors.UNDERLINE + "EMPTY" + Colors.RESET)


        while True:
            query = input("Enter input: ")  
            self.run(lkg, query)


# --- Main Execution ---
if __name__ == "__main__":

    # Setup Logger
    log_filename = "mags_test_run.log"
    # Detect if in Jupyter to adjust console logging (basic for now)
    in_jupyter = 'ipykernel' in sys.modules

    logger = StylizedLogger(name="MAGS_MainTest", level="DEBUG", log_file=None, to_console=True, to_jupyter=in_jupyter)
    logger.info(f"MAGS Pipeline starting. Log file: {log_filename}")
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Current working directory: {os.getcwd()}")


    # Initialize Hyperparameters (can be loaded from a file in a real scenario)
    global_hyperparams = Hyperparameters()
    global_hyperparams.set_param("system_version", "0.1.0-alpha")
    # global_hyperparams.set_param("test_run_timestamp", datetime.now().isoformat())

    global_hyperparams.register_defaults("node", {"default_importance": 1.0, "default_surprise": 0.5, "embedding_dim": 384})
    global_hyperparams.register_defaults("edge", {"default_strength": 0.5})
    global_hyperparams.register_defaults("lkg", {"max_nodes": 1000, "max_edges": 5000, "allow_duplicate_node_content": False})


    test_suite = MAGSGenPipeline(logger, global_hyperparams)
    test_suite.loop()