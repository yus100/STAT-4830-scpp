import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
import os

# Helper function for colored prints (optional, can be reused from PPO.py or simplified)
def print_color(text, color="cyan"):
    colors = {
        "reset": "\033[0m", "black": "\033[30m", "red": "\033[31m", "green": "\033[32m",
        "yellow": "\033[33m", "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
        "white": "\033[37m", "bold_red": "\033[1;31m"
    }
    print(f"{colors.get(color, colors['reset'])}{text}{colors['reset']}")


class PPOInference:
    def __init__(self, checkpoint_base_path: str, checkpoint_name: str = "final", device: str = None):
        """
        Initializes the PPOInference class by loading the model and tokenizer
        from the specified PPO training checkpoint.

        Args:
            checkpoint_base_path (str): The base directory where PPO checkpoints are saved
                                        (e.g., "trained_ppo_model" from PPOHyperparameters.save_path).
            checkpoint_name (str): The specific checkpoint to load (e.g., "final", "10").
            device (str, optional): The device to run inference on ("cuda", "cpu").
                                    If None, uses device from saved hyperparameters or autodetects.
        """
        checkpoint_dir = os.path.join(checkpoint_base_path, f"checkpoint-{checkpoint_name}")
        actor_model_path = os.path.join(checkpoint_dir, "actor")
        hyperparams_path = os.path.join(checkpoint_dir, "ppo_hyperparameters.json")

        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        if not os.path.exists(actor_model_path):
            raise FileNotFoundError(f"Actor model directory not found in checkpoint: {actor_model_path}")
        if not os.path.exists(hyperparams_path):
            raise FileNotFoundError(f"Hyperparameters file not found in checkpoint: {hyperparams_path}")

        print_color(f"Loading PPO hyperparameters from: {hyperparams_path}", "magenta")
        with open(hyperparams_path, 'r') as f:
            self.ppo_params = json.load(f)

        # Determine device
        if device:
            self.device = device
        else:
            self.device = self.ppo_params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        print_color(f"Using device: {self.device}", "magenta")

        print_color(f"Loading tokenizer from: {actor_model_path}", "magenta")
        self.tokenizer = AutoTokenizer.from_pretrained(actor_model_path)
        # Ensure pad_token is set for generation, defaulting to eos_token if not present
        if self.tokenizer.pad_token is None:
            print_color("Tokenizer does not have a pad_token. Setting pad_token = eos_token.", "yellow")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token is None: # Should be caught by PPO script but good check
            raise ValueError("Tokenizer must have an EOS token for generation.")


        print_color(f"Loading actor model from: {actor_model_path}", "magenta")
        self.actor_model = AutoModelForCausalLM.from_pretrained(actor_model_path)
        self.actor_model.to(self.device)
        self.actor_model.eval()  # Set model to evaluation mode

        # --- Setup Default Generation Configuration ---
        # Use pad_token_id from tokenizer, fallback to eos_token_id for GenerationConfig
        resolved_pad_token_id_for_config = self.tokenizer.pad_token_id
        if resolved_pad_token_id_for_config is None: # Should have been set above
            resolved_pad_token_id_for_config = self.tokenizer.eos_token_id

        # PPO training uses sampling, so do_sample=True is a sensible default for inference too.
        default_do_sample = self.ppo_params.get("temperature", 0.7) < 1.0 or \
                            self.ppo_params.get("top_p", 0.9) < 1.0 or \
                            self.ppo_params.get("top_k", 0) > 0

        self.default_generation_config = GenerationConfig(
            max_new_tokens=self.ppo_params.get("max_generated_length", 64),
            temperature=self.ppo_params.get("temperature", 0.7),
            top_k=self.ppo_params.get("top_k", 0),
            top_p=self.ppo_params.get("top_p", 0.9),
            do_sample=default_do_sample, # PPO training usually involves sampling
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=resolved_pad_token_id_for_config
        )
        print_color(f"Default GenerationConfig loaded: eos={self.default_generation_config.eos_token_id}, pad={self.default_generation_config.pad_token_id}, do_sample={self.default_generation_config.do_sample}", "magenta")
        print_color(f"  max_new_tokens={self.default_generation_config.max_new_tokens}, temp={self.default_generation_config.temperature}, top_k={self.default_generation_config.top_k}, top_p={self.default_generation_config.top_p}", "magenta")


    def generate(self, prompt_text: str, **kwargs) -> str:
        """
        Generates a response to the given prompt_text using the loaded actor model.

        Args:
            prompt_text (str): The input text prompt.
            **kwargs: Override default generation parameters.
                      Examples: max_new_tokens, temperature, top_k, top_p, do_sample.

        Returns:
            str: The generated text response (only the newly generated part).
        """
        if not prompt_text:
            print_color("Prompt text is empty. Returning empty string.", "yellow")
            return ""

        print_color(f"\nGenerating response for prompt: '{prompt_text}'", "cyan")

        # Tokenize the input prompt
        # max_length for tokenizer should be based on PPO's max_input_length
        inputs = self.tokenizer(
            prompt_text,
            max_length=self.ppo_params.get("max_input_length", 256),
            truncation=True,
            padding=False,  # No padding for single sequence generation
            return_tensors="pt"
        ).to(self.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask # Will be all 1s if padding=False

        # Prepare generation config for this specific call, starting with defaults
        # and overriding with any kwargs provided.
        current_gen_params = self.default_generation_config.to_dict()

        for key, value in kwargs.items():
            if key in current_gen_params:
                current_gen_params[key] = value
            elif hasattr(GenerationConfig(), key): # Check if it's a valid GenerationConfig attribute
                current_gen_params[key] = value
            else:
                print_color(f"Warning: Generation parameter '{key}' is not a recognized attribute for GenerationConfig.", "yellow")
        
        # Create a new GenerationConfig instance for this specific call
        final_generation_config = GenerationConfig(**current_gen_params)

        # If do_sample was not explicitly passed in kwargs, re-evaluate based on other params
        if 'do_sample' not in kwargs:
            if (final_generation_config.temperature is not None and 0 < final_generation_config.temperature < 1.0) or \
               (final_generation_config.top_p is not None and 0.0 < final_generation_config.top_p < 1.0) or \
               (final_generation_config.top_k is not None and final_generation_config.top_k > 0):
                final_generation_config.do_sample = True
            elif final_generation_config.temperature == 1.0 and \
                 (final_generation_config.top_p is None or final_generation_config.top_p >= 1.0) and \
                 (final_generation_config.top_k is None or final_generation_config.top_k == 0):
                 final_generation_config.do_sample = False # Conditions for greedy

        if kwargs: # If any overrides were passed, print the effective config for this call
            print_color(f"  Effective GenerationConfig for this call: eos={final_generation_config.eos_token_id}, pad={final_generation_config.pad_token_id}, do_sample={final_generation_config.do_sample}", "magenta")
            print_color(f"    max_new_tokens={final_generation_config.max_new_tokens}, temp={final_generation_config.temperature}, top_k={final_generation_config.top_k}, top_p={final_generation_config.top_p}", "magenta")


        with torch.no_grad():  # Ensure no gradients are computed during inference
            # The `generate` method outputs the full sequence (prompt + new tokens)
            generated_ids_full = self.actor_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=final_generation_config
            )

        # Extract only the newly generated tokens
        prompt_length = input_ids.shape[1]
        # generated_ids_full is usually shape [batch_size, sequence_length]
        generated_ids_new_only = generated_ids_full[0, prompt_length:]

        # Decode the generated tokens into text
        response_text = self.tokenizer.decode(generated_ids_new_only, skip_special_tokens=True)

        print_color(f"Generated response: '{response_text.strip()}'", "green")
        return response_text.strip()

if __name__ == "__main__":
    # This is an example of how to use the PPOInference class.
    # It assumes your PPO.py script has successfully run and saved a model.

    # This should match the `save_path` in your PPOHyperparameters
    # (e.g., "trained_ppo_model")
    ppo_model_base_directory = "trained_ppo_model"

    try:
        # Initialize the inferencer, loading the "final" checkpoint
        inferencer = PPOInference(
            checkpoint_base_path=ppo_model_base_directory,
            checkpoint_name="final"
            # device="cpu" # Optionally force CPU
        )

        # Example prompts
        prompts_to_test = [
            # "What is John’s favorite color?",
            # "Record: The team meeting is scheduled for Friday at 3 PM.",
            # "Can you remind me about John's preferences?",
            # "The secret code is 1234. Store this information."
            "What time is the flight at 5PM today?"
        ]

        print_color("\n--- Running Inference with Default Settings ---", "yellow")
        for p_text in prompts_to_test:
            generated_response = inferencer.generate(p_text)
            # The generate method already prints, but you can do more here
            print("-" * 40)

        print_color("\n--- Running Inference with Custom Generation Parameters ---", "yellow")
        custom_prompt = "Write a very short story about a robot learning to dream."
        # Example of overriding generation parameters
        custom_response = inferencer.generate(
            custom_prompt,
            max_new_tokens=50,
            temperature=0.65,
            top_k=40,
            do_sample=True # Explicitly enabling sampling
        )
        print("-" * 40)

        custom_prompt_greedy = "What is the capital of France?"
        # Example of forcing greedy decoding (if not default)
        custom_response_greedy = inferencer.generate(
            custom_prompt_greedy,
            do_sample=False
        )
        print("-" * 40)


    except FileNotFoundError as e:
        print_color(f"ERROR: Could not initialize PPOInference. {e}", "bold_red")
        print_color("Please ensure that you have run the PPO.py training script and that", "yellow")
        print_color(f"a checkpoint (e.g., 'checkpoint-final') exists in '{ppo_model_base_directory}'.", "yellow")
    except Exception as e:
        print_color(f"An unexpected error occurred during inference: {e}", "bold_red")
        import traceback
        traceback.print_exc()