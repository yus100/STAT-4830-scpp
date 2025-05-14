import json
import re
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
import argparse
import os

# --- PPOInference Class ---
# IMPORTANT: This script relies on the PPOInference class.
# It's best if PPOInference is defined in your PPO.py and can be imported.
# If not, you might need to copy the PPOInference class definition here
# or ensure PPO.py is in your PYTHONPATH.

try:
    from PPO import PPOInference # Assuming PPO.py contains a PPOInference class
    from PPO import print_color # Assuming PPO.py also has print_color
except ImportError:
    print("Warning: Could not import PPOInference or print_color from PPO.py.")
    print("Please ensure PPO.py is accessible and these components are defined.")
    print("Using placeholder definitions for script structure.")

    # Placeholder for print_color if not imported
    def print_color(text, color="cyan"):
        # Basic print if PPO's print_color is not available
        print(f"[{color.upper()}] {text}")

    # Placeholder PPOInference class for script structure if not imported.
    # The actual PPOInference class from PPO.py is needed for this script to run.
    class PPOInference:
        def __init__(self, checkpoint_base_path: str, checkpoint_name: str = "final", device: str = None):
            print_color("PPOInference (Placeholder): Initialized. Ensure the real class is used.", "bold_red")
            # Store default_generation_config as an attribute if your real class does
            self.default_generation_config = argparse.Namespace(max_new_tokens=64) # Minimal mock
            # raise NotImplementedError(
            #     "This is a placeholder PPOInference. The actual class from PPO.py must be used."
            # )

        def generate(self, prompt_text: str, **kwargs) -> str:
            print_color(f"PPOInference (Placeholder): Generating for '{prompt_text}'. Returning placeholder.", "bold_red")
            # return "<memory_ask> placeholder response </memory_ask>" # Example placeholder output
            raise NotImplementedError(
                "This is a placeholder PPOInference. The actual class from PPO.py must be used."
            )
# --- End PPOInference ---


@dataclass
class EvalConfig:
    similarity_threshold: float = 0.65  # Min cosine similarity for content match
    tag_match_bonus: float = 0.5       # Score for matching operation tags (ask/write)
    content_similarity_weight: float = 1.0 # Weight for scaled content similarity
    cosine_scaling_factor: float = 2.0 # Power to raise cosine similarity (sim^factor)
                                       # Values > 1 make the score increase more sharply for higher similarities.
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    checkpoint_base_path: str = "trained_ppo_model" # From PPOHyperparameters.save_path
    checkpoint_name: str = "final"
    device: Optional[str] = None # Auto-detect ('cuda' or 'cpu') if None
    # Generation parameters for the model during evaluation
    eval_max_new_tokens: int = 64
    eval_temperature: float = 0.1 # Lower temperature for more deterministic output
    eval_do_sample: bool = False  # Typically False (greedy) for evaluation


# Global cache for embeddings to avoid recomputing for identical content strings
embedding_cache: Dict[str, np.ndarray] = {}


def parse_operations(response_text: str) -> List[Dict[str, str]]:
    """
    Parses a response text into a list of operations.
    Each operation is a dictionary: {"tag": "ask"|"write", "content": "..."}
    """
    pattern = re.compile(r"<memory_(ask|write)>(.*?)</memory_\1>", re.IGNORECASE | re.DOTALL)
    operations = []
    if not isinstance(response_text, str): # Handle cases where response might not be a string
        return operations
    for match in pattern.finditer(response_text):
        tag_type = match.group(1).lower()
        content = match.group(2).strip()
        operations.append({"tag": tag_type, "content": content})
    return operations


class Evaluator:
    def __init__(self, config: EvalConfig, ppo_inference_instance: PPOInference):
        self.config = config
        self.ppo_infer = ppo_inference_instance
        
        effective_device = config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        print_color(f"Initializing Evaluator with embedding model: {config.embedding_model_name} on device: {effective_device}", "magenta")
        self.embedding_model = SentenceTransformer(config.embedding_model_name, device=effective_device)

    def _get_embedding(self, text: str) -> np.ndarray:
        if text in embedding_cache:
            return embedding_cache[text]
        # Ensure text is not empty, SentenceTransformer might error on empty strings list
        if not text: 
            # Return a zero vector of the expected dimensionality
            # Get dimensionality from a dummy encoding if model not yet used
            if not hasattr(self, '_embedding_dim'):
                 try:
                    self._embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                 except Exception:
                    # Fallback if somehow a model doesn't have this method before first encode
                    # (SentenceTransformer models typically do)
                    dummy_emb = self.embedding_model.encode("test", convert_to_numpy=True)
                    self._embedding_dim = dummy_emb.shape[0]
            emb = np.zeros(self._embedding_dim, dtype=np.float32)
        else:
            emb = self.embedding_model.encode(text, convert_to_numpy=True)
        embedding_cache[text] = emb
        return emb

    def _get_embeddings_for_ops(self, ops: List[Dict[str, str]]) -> List[np.ndarray]:
        contents = [op['content'] for op in ops]
        # Batch encoding is more efficient if many contents
        if not hasattr(self, '_embedding_dim') and contents:
            # Determine embedding dimension from the first batch if not known
            first_batch_embeddings = self.embedding_model.encode(contents[:1], convert_to_numpy=True)
            if first_batch_embeddings.ndim > 0 : # check if encoding was successful
                 self._embedding_dim = first_batch_embeddings.shape[1]

        embeddings = []
        for content in contents:
            embeddings.append(self._get_embedding(content))
        return embeddings


    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if emb1.ndim == 1: emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1: emb2 = emb2.reshape(1, -1)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0: # Avoid division by zero for zero vectors
            return 0.0
            
        sim = np.dot(emb1, emb2.T) / (norm1 * norm2)
        return float(sim.item())

    def evaluate_example(self, ground_truth_response: str, generated_response: str) -> Dict[str, int]:
        ops_A = parse_operations(ground_truth_response) # Ground Truth operations
        ops_R = parse_operations(generated_response)   # Model Response operations

        if not ops_A and not ops_R:
            return {"tp": 0, "fp": 0, "fn": 0, "num_gt": 0, "num_pred": 0}
        
        num_ops_A = len(ops_A)
        num_ops_R = len(ops_R)

        if not ops_A: # All predicted are False Positives
            return {"tp": 0, "fp": num_ops_R, "fn": 0, "num_gt": num_ops_A, "num_pred": num_ops_R}
        if not ops_R: # All ground truth are False Negatives
            return {"tp": 0, "fp": 0, "fn": num_ops_A, "num_gt": num_ops_A, "num_pred": num_ops_R}

        emb_A = self._get_embeddings_for_ops(ops_A)
        emb_R = self._get_embeddings_for_ops(ops_R)

        # raw_score_matrix[i, j] = match score between ops_R[i] and ops_A[j]
        # Positive if considered a match, 0 otherwise.
        raw_score_matrix = np.zeros((num_ops_R, num_ops_A))

        for r_idx in range(num_ops_R):
            for a_idx in range(num_ops_A):
                op_r, emb_r = ops_R[r_idx], emb_R[r_idx]
                op_a, emb_a = ops_A[a_idx], emb_A[a_idx]
                
                current_match_score = 0.0
                if op_r["tag"] == op_a["tag"]:
                    sim = self._cosine_similarity(emb_r, emb_a)
                    if sim >= self.config.similarity_threshold:
                        scaled_sim_score = (sim ** self.config.cosine_scaling_factor)
                        current_match_score = self.config.tag_match_bonus + \
                                           (self.config.content_similarity_weight * scaled_sim_score)
                raw_score_matrix[r_idx, a_idx] = current_match_score
        
        # Cost matrix for assignment (minimize negative score to maximize score)
        cost_matrix = -raw_score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        tp_count = 0
        # Iterate through optimal assignments. An assignment is a TP if its score was > 0.
        for r_idx, a_idx in zip(row_ind, col_ind):
            if raw_score_matrix[r_idx, a_idx] > 1e-6: # Check if it was a valid match (score > 0)
                tp_count += 1
        
        fp_count = num_ops_R - tp_count
        fn_count = num_ops_A - tp_count
        
        return {"tp": tp_count, "fp": fp_count, "fn": fn_count, "num_gt": num_ops_A, "num_pred": num_ops_R}

    def run_evaluation(self, dataset_path: str):
        global embedding_cache
        embedding_cache.clear() # Clear cache for a fresh run

        print_color(f"Loading dataset from: {dataset_path}", "blue")
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            examples = dataset.get("examples", [])
            if not examples:
                print_color(f"No 'examples' found in {dataset_path} or the list is empty.", "bold_red")
                return None
        except FileNotFoundError:
            print_color(f"ERROR: Dataset file not found: {dataset_path}", "bold_red")
            return None
        except json.JSONDecodeError as e:
            print_color(f"ERROR: Could not decode JSON from {dataset_path}: {e}", "bold_red")
            return None

        total_tp, total_fp, total_fn = 0, 0, 0
        total_gt_ops, total_pred_ops = 0, 0
        
        generation_args = {
            "max_new_tokens": self.config.eval_max_new_tokens,
            "temperature": self.config.eval_temperature,
            "do_sample": self.config.eval_do_sample,
        }
        print_color(f"Using PPO model generation arguments for evaluation: {generation_args}", "magenta")

        for i, example in enumerate(examples):
            prompt = example.get("input", "")
            ground_truth_response = example.get("response", "")
            
            if not prompt:
                print_color(f"Skipping example {i+1} due to empty prompt.", "yellow")
                continue

            print_color(f"\nEvaluating example {i+1}/{len(examples)}", "cyan")
            print_color(f"  Prompt: '{prompt[:150].replace(os.linesep, ' ')}...'", "default") # Using default color for less noise
            print_color(f"  Ground Truth Ops: {ground_truth_response[:150].replace(os.linesep, ' ')}...", "yellow")

            try:
                generated_response = self.ppo_infer.generate(prompt, **generation_args)
            except Exception as e:
                print_color(f"  Error during model generation for example {i+1}: {e}", "bold_red")
                generated_response = "" # Treat as empty generation on error


            counts = self.evaluate_example(ground_truth_response, generated_response)
            
            total_tp += counts["tp"]
            total_fp += counts["fp"]
            total_fn += counts["fn"]
            total_gt_ops += counts["num_gt"]
            total_pred_ops += counts["num_pred"]

            print_color(f"  Stats: TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']} (GT ops: {counts['num_gt']}, Pred ops: {counts['num_pred']})", "green")

        # Micro-averaged Precision, Recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print_color("\n--- Overall Evaluation Results ---", "blue")
        print_color(f"Total Ground Truth Operations: {total_gt_ops}", "yellow")
        print_color(f"Total Predicted Operations:    {total_pred_ops}", "yellow")
        print_color(f"Total True Positives (TP):     {total_tp}", "green")
        print_color(f"Total False Positives (FP):    {total_fp}", "red")
        print_color(f"Total False Negatives (FN):    {total_fn}", "yellow")
        print_color(f"Precision (micro-avg):         {precision:.4f}", "cyan")
        print_color(f"Recall (micro-avg):            {recall:.4f}", "cyan")
        print_color(f"F1-Score (micro-avg):          {f1_score:.4f}", "cyan")
        
        results = {"precision": precision, "recall": recall, "f1_score": f1_score,
                   "tp": total_tp, "fp": total_fp, "fn": total_fn,
                   "total_gt_ops": total_gt_ops, "total_pred_ops": total_pred_ops}
        print_color(f"\nResults dictionary: {results}", "magenta")
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PPO-trained model.")
    parser.add_argument("--dataset_path", type=str, default="data/d1.json", help="Path to the JSON dataset file.")
    parser.add_argument("--checkpoint_base_path", type=str, default="trained_ppo_model", help="Base PPO checkpoint directory.")
    parser.add_argument("--checkpoint_name", type=str, default="final", help="Specific checkpoint name (e.g., 'final', '10').")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', or None for auto).")
    
    # EvalConfig parameters
    parser.add_argument("--similarity_threshold", type=float, default=EvalConfig.similarity_threshold)
    parser.add_argument("--tag_match_bonus", type=float, default=EvalConfig.tag_match_bonus)
    parser.add_argument("--content_similarity_weight", type=float, default=EvalConfig.content_similarity_weight)
    parser.add_argument("--cosine_scaling_factor", type=float, default=EvalConfig.cosine_scaling_factor)
    parser.add_argument("--embedding_model_name", type=str, default=EvalConfig.embedding_model_name)
    
    # Generation parameters for evaluation
    parser.add_argument("--eval_max_new_tokens", type=int, default=EvalConfig.eval_max_new_tokens)
    parser.add_argument("--eval_temperature", type=float, default=EvalConfig.eval_temperature)
    parser.add_argument("--eval_do_sample", type=lambda x: (str(x).lower() == 'true'), default=EvalConfig.eval_do_sample)


    args = parser.parse_args()

    eval_config = EvalConfig(
        similarity_threshold=args.similarity_threshold,
        tag_match_bonus=args.tag_match_bonus,
        content_similarity_weight=args.content_similarity_weight,
        cosine_scaling_factor=args.cosine_scaling_factor,
        embedding_model_name=args.embedding_model_name,
        checkpoint_base_path=args.checkpoint_base_path,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        eval_max_new_tokens=args.eval_max_new_tokens,
        eval_temperature=args.eval_temperature,
        eval_do_sample=args.eval_do_sample
    )
    
    # Ensure torch is imported for device check if PPOInference placeholder is used.
    # The real PPOInference will handle torch internally.
    import torch 

    # Initialize PPOInference (CRITICAL: actual class must be available)
    ppo_inference_instance = None
    try:
        # Re-attempt import here in case it was a placeholder initially
        from Inference import PPOInference as ActualPPOInference
        ppo_inference_instance = ActualPPOInference(
            checkpoint_base_path=eval_config.checkpoint_base_path,
            checkpoint_name=eval_config.checkpoint_name,
            device=eval_config.device  # Pass the device from config
        )
    except ImportError:
        print_color("CRITICAL ERROR: Could not import PPOInference from PPO.py.", "bold_red")
        print_color("Evaluation cannot proceed without a working PPOInference class.", "bold_red")
        print_color("Please ensure PPO.py is in your Python path or copy the PPOInference class definition here.", "bold_red")
        return
    except FileNotFoundError as e:
        print_color(f"CRITICAL ERROR: PPOInference initialization failed (model/checkpoint files not found?): {e}", "bold_red")
        return
    except Exception as e:
        print_color(f"CRITICAL ERROR: Unexpected error initializing PPOInference: {e}", "bold_red")
        import traceback
        traceback.print_exc()
        return

    evaluator = Evaluator(config=eval_config, ppo_inference_instance=ppo_inference_instance)
    evaluator.run_evaluation(dataset_path=args.dataset_path)

if __name__ == "__main__":
    main()