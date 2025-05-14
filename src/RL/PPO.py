import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from tqdm import tqdm
import numpy as np
import json
import random
import os
import re
from difflib import SequenceMatcher

# Helper function for colored prints (optional)
def print_color(text, color="cyan"):
    colors = {
        "reset": "\033[0m", "black": "\033[30m", "red": "\033[31m", "green": "\033[32m",
        "yellow": "\033[33m", "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
        "white": "\033[37m", "bold_red": "\033[1;31m"
    }
    print(f"{colors.get(color, colors['cyan'])}{text}{colors['reset']}")

class PPOHyperparameters:
    def __init__(self):
        # --- Model & Tokenizer ---
        self.model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # or "gpt2", "meta-llama/Llama-2-7b-hf"
        self.tokenizer_name: str = None # If None, uses model_name
        self.max_input_length: int = 256 # Max length for input prompt tokens
        self.max_generated_length: int = 64  # Max length for generated response tokens

        # --- Data ---
        self.data_file: str = "data/d1.json" # Path to your JSON data file
        self.val_split_ratio: float = 0.1 # Percentage of data for validation (not used in PPO directly, but good practice)

        # --- PPO Core ---
        self.ppo_epochs: int = 4 # Number of optimization epochs per batch of experience
        self.num_rollouts: int = 100 # Total number of rollouts (outer training loops)
        self.num_episodes_per_rollout: int = 64 # Number of episodes (prompts) to collect in each rollout
                                                # This acts as the batch size for experience collection
        self.mini_batch_size: int = 8 # Mini-batch size for PPO updates (must be <= num_episodes_per_rollout)
        self.gamma: float = 0.99  # Discount factor for future rewards
        self.lambda_gae: float = 0.95  # GAE lambda parameter
        self.clip_epsilon: float = 0.2  # PPO clipping parameter
        self.vf_coef: float = 0.5  # Value function loss coefficient
        self.entropy_coef: float = 0.01 # Entropy bonus coefficient
        self.learning_rate_actor: float = 1e-5
        self.learning_rate_critic: float = 1e-5 # Can be same or different from actor
        self.adam_epsilon: float = 1e-8 # AdamW epsilon
        self.max_grad_norm: float = 0.5 # Gradient clipping max norm

        # --- Generation ---
        self.temperature: float = 0.7 # Temperature for sampling during generation (exploration)
        self.top_k: int = 0 # if >0, filter for top_k tokens
        self.top_p: float = 0.9 # if >0 and <1, nucleus sampling

        # --- Training & System ---
        self.seed: int = 42
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_path: str = "trained_ppo_model"
        self.save_interval: int = 10 # Save model every N rollouts
        self.log_interval: int = 1 # Log stats every N rollouts

        # --- Reward specific (example) ---
        self.reward_tag_match_bonus: float = 0.5
        self.reward_content_similarity_scale: float = 0.5
        self.reward_malformed_penalty: float = -0.2

    def print_params(self):
        print_color("--- PPO Hyperparameters ---", "yellow")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print_color("--------------------------", "yellow")

# Custom Dataset
class PromptDataset(Dataset):
    def __init__(self, data_file_path, tokenizer, max_input_length):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.examples = []

        print_color(f"Loading data from: {data_file_path}", "green")
        try:
            with open(data_file_path, 'r') as f:
                data = json.load(f)
            self.examples = data.get("examples", [])
            if not self.examples:
                raise ValueError("No 'examples' found in JSON file or 'examples' list is empty.")
            print_color(f"Loaded {len(self.examples)} examples.", "green")
        except FileNotFoundError:
            print_color(f"ERROR: Data file not found at {data_file_path}", "bold_red")
            raise
        except json.JSONDecodeError:
            print_color(f"ERROR: Could not decode JSON from {data_file_path}", "bold_red")
            raise
        except ValueError as e:
            print_color(f"ERROR: {e}", "bold_red")
            raise

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["input"]
        target_response_text = example["response"] # This is the "gold" response for reward calculation

        tokenized_input = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length", # Pad to max_input_length for batching
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized_input.input_ids.squeeze(0),
            "attention_mask": tokenized_input.attention_mask.squeeze(0),
            "input_text": input_text, # Keep original text for generation context
            "target_response_text": target_response_text
        }

# Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, model_name, tokenizer_eos_token_id):
        super().__init__()
        print_color(f"Initializing ActorCritic model with base: {model_name}", "magenta")
        self.config = AutoConfig.from_pretrained(model_name)
        self.actor_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add a value head for the Critic
        # The value head takes the hidden states of the transformer and outputs a scalar value
        # We use the hidden size of the last layer of the transformer
        self.value_head = nn.Linear(self.config.hidden_size, 1)
        self.tokenizer_eos_token_id = tokenizer_eos_token_id
        print_color("ActorCritic model initialized.", "magenta")

    def forward(self, input_ids, attention_mask, generate=False, gen_kwargs=None):
        # Actor part (language model)
        # Get logits for next token prediction
        transformer_outputs = self.actor_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True # Ensure hidden states are outputted
        )
        # Logits for token prediction
        # The lm_head is usually applied automatically by AutoModelForCausalLM when calling its forward directly for loss
        # but here we need raw logits from the base model output for generation
        # and the value from the hidden state
        
        # For generation, we might use the actor_model.generate() method outside this forward if simpler
        # For getting log_probs of an existing sequence, we need lm_head on hidden_states
        
        # Critic part (value function)
        # Use the hidden state of the last token (or average) for value estimation
        # For causal LMs, the last hidden state of the input sequence is a good representation
        last_hidden_state = transformer_outputs.last_hidden_state
        
        # Value: use the hidden state of the *last input token*
        # Find the index of the last non-padding token for each sequence in the batch
        if input_ids.ndim == 1: # if single sequence
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            last_hidden_state = last_hidden_state.unsqueeze(0)


        sequence_lengths = torch.ne(input_ids, self.tokenizer_eos_token_id).sum(-1) -1 # Get actual length
        sequence_lengths = sequence_lengths.clamp(min=0) # Ensure not negative if all are PAD
        
        # Gather the hidden state for the last token of each sequence
        # For value, we use the representation of the input prompt.
        # A common choice is the hidden state of the *last token of the input prompt*.
        # (Batch_size, seq_len, hidden_size) -> (Batch_size, hidden_size)
        # value_hidden_state = last_hidden_state[torch.arange(last_hidden_state.shape[0]), sequence_lengths]
        # A simpler approach sometimes used is to average pool or take the first token's ([CLS] equivalent for encoders) state.
        # For decoder-only models, the last token's hidden state before generation starts is meaningful.
        value_hidden_state = last_hidden_state[:, -1, :] # Using the very last hidden state from input
        value = self.value_head(value_hidden_state).squeeze(-1) # (Batch_size)

        if generate and gen_kwargs is not None:
            # This part is tricky if we want to get log_probs and generated sequence simultaneously
            # actor_model.generate will give token_ids
            # For PPO, we need log_probs of the *actions taken*
            # It's often easier to separate generation and log_prob calculation
            # Here, we'll just return the value, generation is handled outside
            return value # Generation logic needs to be separate to get log_probs correctly
        else:
            # If not generating, but evaluating an existing sequence (e.g. input_ids includes prompt + response)
            # then we need logits for calculating log_probs of the response part.
            # This is for the PPO update phase.
            logits = self.actor_model.lm_head(last_hidden_state) # (Batch_size, seq_len, vocab_size)
            return logits, value


    def get_action_and_value(self, input_ids_prompt, attention_mask_prompt, generation_config, device):
        """
        Generates an action (response sequence) and estimates the value of the state (prompt).
        Also calculates the log probability of the generated action.
        """
        input_ids_prompt = input_ids_prompt.to(device)
        attention_mask_prompt = attention_mask_prompt.to(device)

        # Get value for the prompt state
        self.actor_model.eval() # Ensure model is in eval mode for generation consistency if using dropout etc.
        self.value_head.eval()
        with torch.no_grad():
            transformer_outputs = self.actor_model.base_model(
                input_ids=input_ids_prompt,
                attention_mask=attention_mask_prompt,
                output_hidden_states=True
            )
            last_hidden_state_prompt = transformer_outputs.last_hidden_state
            # value_hidden_state_prompt = last_hidden_state_prompt[torch.arange(last_hidden_state_prompt.shape[0]), torch.ne(input_ids_prompt, self.tokenizer_eos_token_id).sum(-1) -1]
            value_hidden_state_prompt = last_hidden_state_prompt[:, -1, :]
            value = self.value_head(value_hidden_state_prompt).squeeze(-1) # (Batch_size)

        # Generate action (response)
        self.actor_model.train() # Switch back to train for PPO updates later, or keep eval if reference model
        self.value_head.train()

        if generation_config.pad_token_id is None:
            # This check should ideally be handled by the PPOTrainer's generation_config setup
            # but as a safeguard:
            generation_config.pad_token_id = self.tokenizer_eos_token_id
            print_color(f"Warning in get_action_and_value: generation_config.pad_token_id was None. Set to {generation_config.pad_token_id}", "yellow")


        generated_outputs = self.actor_model.generate(
            input_ids=input_ids_prompt,
            attention_mask=attention_mask_prompt,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False # output_scores=True can be very memory intensive for log_prob calculation here.
                               # It's generally easier to re-evaluate the generated sequence.
        )
        generated_sequences_full = generated_outputs.sequences
        generated_sequences_action_only = generated_sequences_full[:, input_ids_prompt.shape[1]:]

        # Calculate log_probs for the generated_sequences_action_only
        full_sequence_ids = generated_sequences_full
        action_attention_mask_for_logprobs = (generated_sequences_action_only != generation_config.pad_token_id).long().to(device)
        full_attention_mask_for_logprobs = torch.cat([attention_mask_prompt, action_attention_mask_for_logprobs], dim=1)
        
        # Ensure the full_attention_mask_for_logprobs does not exceed the length of full_sequence_ids if any truncation happened unexpectedly
        # Or if generated_sequences_action_only was shorter than max_generated_length and not padded by .generate() to full combined length for model input
        if full_attention_mask_for_logprobs.shape[1] > full_sequence_ids.shape[1]:
            full_attention_mask_for_logprobs = full_attention_mask_for_logprobs[:, :full_sequence_ids.shape[1]]


        model_outputs_for_logprobs = self.actor_model(
            input_ids=full_sequence_ids,
            attention_mask=full_attention_mask_for_logprobs
        )
        all_logits = model_outputs_for_logprobs.logits # B x (prompt_len + gen_len) x Vocab_size

        prompt_len = input_ids_prompt.shape[1]
        action_len = generated_sequences_action_only.shape[1]

        # Logits for the action part:
        # logit at index `t` is for predicting token at `t+1`.
        # So, for `action_token[0]` (which is `full_sequence_ids[prompt_len]`), we need `all_logits[:, prompt_len-1, :]`.
        # For `action_token[k]` (which is `full_sequence_ids[prompt_len+k]`), we need `all_logits[:, prompt_len+k-1, :]`.
        # The slice `all_logits[:, prompt_len - 1 : prompt_len - 1 + action_len, :]` captures these.
        action_logits = all_logits[:, prompt_len - 1 : prompt_len - 1 + action_len, :]

        log_probs_dist = F.log_softmax(action_logits, dim=-1)
        # `generated_sequences_action_only` has shape (B, action_len)
        # `log_probs_dist` has shape (B, action_len, Vocab_size)
        # We want to gather the log_probs for the specific tokens in `generated_sequences_action_only`
        log_probs = torch.gather(log_probs_dist, 2,
                                 generated_sequences_action_only.unsqueeze(-1)).squeeze(-1) # Shape: (B, action_len)

        # Mask out log_probs for padding tokens in the action
        # This uses `action_attention_mask_for_logprobs` which is already on the correct device
        action_mask = action_attention_mask_for_logprobs.float()
        sum_log_probs = (log_probs * action_mask).sum(dim=1) # Sum log_probs for the sequence

        return generated_sequences_action_only.cpu(), sum_log_probs.cpu().detach(), value.cpu().detach()

class PPOTrainer:
    def __init__(self, model: ActorCritic, tokenizer, params: PPOHyperparameters, dataset: PromptDataset):
        self.model = model
        self.ref_model = None # For KL divergence, if used. For PPO ratio, this is implicitly the model at experience collection time.
        self.tokenizer = tokenizer
        self.params = params
        self.dataset = dataset # Full dataset, we'll sample from it for rollouts

        self.optimizer_actor = AdamW(self.model.actor_model.parameters(), lr=params.learning_rate_actor, eps=params.adam_epsilon)
        self.optimizer_critic = AdamW(self.model.value_head.parameters(), lr=params.learning_rate_critic, eps=params.adam_epsilon)
        
        self.model.to(params.device)

        # # Generation configuration
        # self.generation_config = AutoConfig.from_pretrained(params.model_name)
        # # Update gen config from params
        # self.generation_config.max_new_tokens = params.max_generated_length
        # self.generation_config.temperature = params.temperature
        # self.generation_config.top_k = params.top_k
        # self.generation_config.top_p = params.top_p
        # self.generation_config.do_sample = True # Important for exploration
        # self.generation_config.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.generation_config.eos_token_id
        # self.generation_config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id # Often set pad = eos for GPT like models

        # if self.generation_config.pad_token_id is None:
        #     print_color("Warning: pad_token_id is None in generation config. Setting to eos_token_id.", "yellow")
        #     self.generation_config.pad_token_id = self.generation_config.eos_token_id

        # --- Generation configuration ---
        # Create a GenerationConfig object.
        # Load any defaults from the model_name, then override with PPOHyperparameters.

        # Determine pad_token_id: use tokenizer's if available, else use eos_token_id
        resolved_pad_token_id = self.tokenizer.pad_token_id
        if resolved_pad_token_id is None:
            print_color(f"Warning: tokenizer.pad_token_id is None. Using tokenizer.eos_token_id ({self.tokenizer.eos_token_id}) as pad_token_id for generation.", "yellow")
            resolved_pad_token_id = self.tokenizer.eos_token_id

        if self.tokenizer.eos_token_id is None:
            # This should be caught by tokenizer setup in main(), but good to double check
            raise ValueError("EOS token ID is not set in the tokenizer, which is required for generation_config.")
        if resolved_pad_token_id is None: # Should not happen if eos_token_id is set
            raise ValueError("pad_token_id could not be resolved for generation_config.")

        self.generation_config = GenerationConfig.from_pretrained(
            params.model_name, # Base settings from model if available
            # Override with parameters from PPOHyperparameters
            max_new_tokens=params.max_generated_length,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            do_sample=True, # Crucial for temperature, top_k, top_p to take effect
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=resolved_pad_token_id
        )

        print_color(f"Initialized GenerationConfig: eos_token_id={self.generation_config.eos_token_id}, pad_token_id={self.generation_config.pad_token_id}", "magenta")
        # No further dynamic modification of self.generation_config.pad_token_id is needed in get_action_and_value


    def parse_generated_response(self, text):
        match_ask = re.search(r"<memory_ask>(.*?)</memory_ask>", text, re.DOTALL)
        match_write = re.search(r"<memory_write>(.*?)</memory_write>", text, re.DOTALL)
        if match_ask:
            return "memory_ask", match_ask.group(1).strip()
        elif match_write:
            return "memory_write", match_write.group(1).strip()
        return None, None

    def compute_reward(self, generated_text, target_text):
        # Example reward function:
        # +0.5 for correct tag (memory_ask/memory_write)
        # +0.5 for content similarity (scaled)
        # -0.2 if malformed (no tag)

        gen_tag, gen_content = self.parse_generated_response(generated_text)
        target_tag, target_content = self.parse_generated_response(target_text)

        reward = 0.0

        if gen_tag is None:
            return self.params.reward_malformed_penalty # Penalize malformed generation

        if gen_tag == target_tag and target_tag is not None:
            reward += self.params.reward_tag_match_bonus
            if gen_content is not None and target_content is not None:
                similarity = SequenceMatcher(None, gen_content, target_content).ratio()
                reward += self.params.reward_content_similarity_scale * similarity
        # Optional: Penalize if tag is correct but content is missing, or if tags mismatch
        elif gen_tag != target_tag and target_tag is not None:
             reward -= 0.1 # Small penalty for wrong tag type

        return reward

    def collect_rollout_experiences(self):
        print_color(f"\nCollecting {self.params.num_episodes_per_rollout} experiences...", "cyan")
        
        rollout_buffer = {
            "prompt_input_ids": [], "prompt_attention_masks": [],
            "action_token_ids": [], "action_log_probs": [],
            "rewards": [], "values": [], "advantages": [], "returns": [],
            "decoded_prompts": [], "decoded_actions": [], "decoded_targets": []
        }
        
        self.model.eval() # Set model to eval for generation

        num_collected = 0
        # Use a DataLoader to sample from the dataset without replacement for this rollout batch
        # If num_episodes_per_rollout > len(dataset), it will loop.
        # Better to sample indices randomly.
        
        dataset_indices = random.sample(range(len(self.dataset)), k=min(self.params.num_episodes_per_rollout, len(self.dataset)))
        if self.params.num_episodes_per_rollout > len(self.dataset):
            print_color(f"Warning: num_episodes_per_rollout ({self.params.num_episodes_per_rollout}) > dataset size ({len(self.dataset)}). Sampling with replacement essentially.", "yellow")
            dataset_indices = random.choices(range(len(self.dataset)), k=self.params.num_episodes_per_rollout)


        batch_data = [self.dataset[i] for i in dataset_indices]
        
        # Collate manually for this specific use case
        # This simple collation assumes all inputs are already padded to max_input_length by the dataset
        prompt_input_ids_batch = torch.stack([item["input_ids"] for item in batch_data]).to(self.params.device)
        prompt_attention_mask_batch = torch.stack([item["attention_mask"] for item in batch_data]).to(self.params.device)
        input_texts_batch = [item["input_text"] for item in batch_data]
        target_response_texts_batch = [item["target_response_text"] for item in batch_data]

        # Generate actions and get values
        # This needs to be done per prompt if model.generate can't batch diverse length inputs easily
        # or if `get_action_and_value` is not batched (current one is designed for batch)
        
        action_token_ids_batch, action_log_probs_batch, values_batch = self.model.get_action_and_value(
            prompt_input_ids_batch,
            prompt_attention_mask_batch,
            self.generation_config,
            self.params.device
        )
        # Ensure outputs are on CPU for storage in buffer and decoding
        action_token_ids_batch = action_token_ids_batch.cpu()
        action_log_probs_batch = action_log_probs_batch.cpu().detach()
        values_batch = values_batch.cpu().detach()


        for i in range(prompt_input_ids_batch.shape[0]):
            prompt_ids = prompt_input_ids_batch[i].cpu()
            prompt_mask = prompt_attention_mask_batch[i].cpu()
            action_ids = action_token_ids_batch[i] # Already on CPU
            log_prob = action_log_probs_batch[i]  # Already on CPU
            value = values_batch[i]               # Already on CPU

            input_text = input_texts_batch[i]
            target_response_text = target_response_texts_batch[i]

            # Decode action for reward calculation and logging
            # Ensure skip_special_tokens=True if special tokens are not part of the "content"
            decoded_action = self.tokenizer.decode(action_ids, skip_special_tokens=True).strip()
            
            reward = self.compute_reward(decoded_action, target_response_text)

            rollout_buffer["prompt_input_ids"].append(prompt_ids)
            rollout_buffer["prompt_attention_masks"].append(prompt_mask)
            rollout_buffer["action_token_ids"].append(action_ids) # Store generated token ids
            rollout_buffer["action_log_probs"].append(log_prob)
            rollout_buffer["rewards"].append(torch.tensor(reward, dtype=torch.float32))
            rollout_buffer["values"].append(value)
            
            rollout_buffer["decoded_prompts"].append(input_text)
            rollout_buffer["decoded_actions"].append(decoded_action)
            rollout_buffer["decoded_targets"].append(target_response_text)

            num_collected += 1
            if num_collected >= self.params.num_episodes_per_rollout:
                break
        
        if num_collected == 0:
            print_color("No experiences collected. This should not happen if dataset is not empty.", "bold_red")
            return None
            
        print_color(f"Collected {num_collected} experiences. Example:", "green")
        print(f"  Prompt: {rollout_buffer['decoded_prompts'][0][:100]}...")
        print(f"  Target: {rollout_buffer['decoded_targets'][0][:100]}...")
        print(f"  Generated: {rollout_buffer['decoded_actions'][0][:100]}...")
        print(f"  Reward: {rollout_buffer['rewards'][0].item():.2f}, Value: {rollout_buffer['values'][0].item():.2f}")

        # Compute GAE (Generalized Advantage Estimation) and returns
        # This part assumes all collected experiences form one long trajectory for GAE calculation,
        # or are treated as batch of single-step episodes if reward is immediate.
        # For LLM PPO, typically each prompt-response is an episode.
        
        rewards = torch.stack(rollout_buffer["rewards"])
        values = torch.stack(rollout_buffer["values"])
        
        # For single step episodes (each prompt-response is an episode)
        # advantages = rewards - values
        # returns = rewards (or rewards + gamma * V(s_next) if there was a next state)
        # In this LLM setup, response generation is one "action". The reward is for this full action.
        # There isn't a V(s_next) after a full response unless we model turn-taking dialogue.
        # So, for simplicity, treat V(s_next) = 0 for terminal states (end of response).
        # GAE calculation needs to be adjusted if episodes are multi-step.
        # Here, each (prompt, response) is an episode. The "value" is V(prompt). Reward is R(prompt, response).
        # Advantage A(prompt, response) = R(prompt, response) - V(prompt).
        # The 'return' for updating V(prompt) would be R(prompt, response).

        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        # Assuming values are V(s_t) and rewards are r_t for taking action a_t at s_t.
        # If these are single step episodes, then delta_t = r_t - V(s_t) (as V(s_{t+1}) = 0)
        # For GAE: A_t = sum_{l=0}^{T-t-1} (gamma*lambda)^l * delta_{t+l}
        # Here T=1 (one step: prompt -> response). So A_t = delta_t = r_t - V(s_t)
        # This simplification is common in "RLHF from scratch" tutorials for LLMs.
        # If we wanted true multi-step GAE *within* a generation, rewards would be per token.
        # But here, reward is per full response.
        
        # For a sequence of (s,a,r) from one episode:
        # rewards = [r_0, r_1, ..., r_{T-1}]
        # values = [V(s_0), V(s_1), ..., V(s_{T-1})]
        # If V(s_T) is the value of terminal state (usually 0)
        # deltas = [r_t + gamma * values_{t+1} - values_t]
        # For our case, each prompt-response is an episode. So V(s_{t+1}) is effectively 0 (terminal).
        # The "value" we have is V(prompt). The "reward" is for the (prompt, generated_response) pair.
        # So, advantage = reward - value(prompt)
        advantages = rewards - values
        returns = rewards # Target for value function is just the immediate reward for these single-step episodes
        
        rollout_buffer["advantages"] = advantages
        rollout_buffer["returns"] = returns

        # Convert lists of tensors to single tensors
        rollout_buffer["prompt_input_ids"] = torch.stack(rollout_buffer["prompt_input_ids"])
        rollout_buffer["prompt_attention_masks"] = torch.stack(rollout_buffer["prompt_attention_masks"])
        rollout_buffer["action_token_ids"] = torch.stack(rollout_buffer["action_token_ids"])
        rollout_buffer["action_log_probs"] = torch.stack(rollout_buffer["action_log_probs"])
        rollout_buffer["rewards"] = rewards # already a tensor
        rollout_buffer["values"] = values   # already a tensor

        return rollout_buffer

    def update_policy(self, rollout_buffer):
        print_color(f"Updating policy for {self.params.ppo_epochs} PPO epochs...", "cyan")
        self.model.train() # Set model to train for updates

        num_samples = rollout_buffer["prompt_input_ids"].size(0)
        if num_samples == 0:
            print_color("No samples in rollout buffer to update policy. Skipping update.", "yellow")
            return

        if self.params.mini_batch_size > num_samples:
            print_color(f"Warning: mini_batch_size ({self.params.mini_batch_size}) > num_samples ({num_samples}). Using num_samples as mini_batch_size.", "yellow")
            # It's better to ensure mini_batch_size is not changed if it's a parameter,
            # but handle smaller last batch in the loop.
            # The loop already handles the last batch potentially being smaller.
            # self.params.mini_batch_size = num_samples # Avoid changing params object here


        # Normalize advantages (optional but often helpful)
        advantages = rollout_buffer["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = rollout_buffer["action_log_probs"].detach()
        old_values = rollout_buffer["values"].detach() # V(prompt) from rollout
        returns = rollout_buffer["returns"].detach()    # Target for V(prompt)

        for ppo_epoch_num in range(self.params.ppo_epochs): # Added for clarity in logging if needed
            permutation = torch.randperm(num_samples)

            for start_idx in range(0, num_samples, self.params.mini_batch_size):
                end_idx = start_idx + self.params.mini_batch_size
                if end_idx > num_samples:
                    end_idx = num_samples

                if start_idx == end_idx: continue

                mini_batch_indices = permutation[start_idx:end_idx]

                mb_prompt_ids = rollout_buffer["prompt_input_ids"][mini_batch_indices].to(self.params.device)
                mb_prompt_masks = rollout_buffer["prompt_attention_masks"][mini_batch_indices].to(self.params.device)
                mb_action_ids = rollout_buffer["action_token_ids"][mini_batch_indices].to(self.params.device)

                mb_old_log_probs = old_log_probs[mini_batch_indices].to(self.params.device)
                mb_advantages = advantages[mini_batch_indices].to(self.params.device)
                mb_returns = returns[mini_batch_indices].to(self.params.device)
                # mb_old_values = old_values[mini_batch_indices].to(self.params.device) # For clipped value loss if used

                # --- Actor Loss ---
                full_input_ids = torch.cat([mb_prompt_ids, mb_action_ids], dim=1)
                action_attention_mask = (mb_action_ids != self.tokenizer.pad_token_id).long().to(self.params.device)
                
                # Ensure full_attention_mask does not exceed model's max sequence length if full_input_ids is truncated
                # However, full_input_ids should already respect max_input_length + max_generated_length
                full_attention_mask = torch.cat([mb_prompt_masks, action_attention_mask], dim=1)


                # This model call uses ActorCritic.forward
                logits, _ = self.model( # current_values_of_full_seq is not used here for actor/critic loss
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask
                )

                prompt_len = mb_prompt_ids.shape[1]
                action_len = mb_action_ids.shape[1]
                
                # Slice logits for the action part
                # logit[t] is for token[t+1]. For action_token[0] (at full_input_ids[prompt_len]), use logit[prompt_len-1]
                action_logits = logits[:, prompt_len - 1 : prompt_len - 1 + action_len, :]
                
                new_log_probs_dist = F.log_softmax(action_logits, dim=-1)
                gathered_new_log_probs = torch.gather(new_log_probs_dist, 2, mb_action_ids.unsqueeze(-1)).squeeze(-1)
                
                # Mask for padding tokens in the action
                action_value_mask = (mb_action_ids != self.tokenizer.pad_token_id).float()
                sum_new_log_probs = (gathered_new_log_probs * action_value_mask).sum(dim=1)

                log_ratio = sum_new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.params.clip_epsilon, 1.0 + self.params.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Critic Loss ---
                # Get V(prompt) using current value head and current (potentially updated if epoch > 0) base model
                # This ensures the value estimate is based on the most current state representation ability of the base model
                # if the base model itself is also being fine-tuned.
                with torch.no_grad(): # Temporarily set no_grad for this part if base_model is frozen or to be safe
                                     # However, for the value head to learn, it needs gradients from here.
                                     # The key is .detach() before value_head if base_model has been updated by actor.
                    transformer_outputs_prompt = self.model.actor_model.base_model(
                        input_ids=mb_prompt_ids,
                        attention_mask=mb_prompt_masks,
                        output_hidden_states=True
                    )
                    last_hidden_state_prompt = transformer_outputs_prompt.last_hidden_state
                
                value_hidden_state_prompt = last_hidden_state_prompt[:, -1, :]
                # detach() here is crucial because actor_model.base_model parameters will be updated
                # by optimizer_actor before optimizer_critic.step() for value_head is called.
                v_s_prompt = self.model.value_head(value_hidden_state_prompt.detach()).squeeze(-1)


                critic_loss = F.mse_loss(v_s_prompt, mb_returns) # Using unclipped for simplicity

                # --- Entropy Bonus (for actor) ---
                probs_dist = F.softmax(action_logits, dim=-1) # action_logits from current policy
                log_probs_dist_entropy = F.log_softmax(action_logits, dim=-1)
                entropy_per_token = -(probs_dist * log_probs_dist_entropy).sum(dim=-1) # B x ActionSeqLen
                
                masked_entropy = entropy_per_token * action_value_mask
                avg_entropy_per_sequence = masked_entropy.sum(dim=1) / action_value_mask.sum(dim=1).clamp(min=1)
                entropy_bonus = avg_entropy_per_sequence.mean()

                # Total Loss for actor
                total_loss_actor = actor_loss - self.params.entropy_coef * entropy_bonus
                # Total Loss for critic (already calculated as critic_loss)
                # The coefficient self.params.vf_coef is applied when combining if there was a single optimizer
                # Here, since optimizers are separate, we can scale it before backward or just use it as is.
                # Let's assume vf_coef scales the critic_loss directly if they were combined.
                # For separate optimizers, often the raw critic_loss is used. Let's apply it.
                total_loss_critic = self.params.vf_coef * critic_loss


                # Optimization step for actor
                self.optimizer_actor.zero_grad()
                total_loss_actor.backward()
                if self.params.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.actor_model.parameters(), self.params.max_grad_norm)
                self.optimizer_actor.step()

                # Optimization step for critic
                self.optimizer_critic.zero_grad()
                total_loss_critic.backward()
                if self.params.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.value_head.parameters(), self.params.max_grad_norm)
                self.optimizer_critic.step()

        # Log losses from the last mini-batch of the last PPO epoch
        if num_samples > 0 : # Ensure there were updates
             print_color(f"  PPO Update Stats (last mini-batch of epoch {ppo_epoch_num+1}): Actor Loss={actor_loss.item():.4f}, Critic Loss={critic_loss.item():.4f}, Entropy={entropy_bonus.item():.4f}", "green")
        else:
             print_color("  PPO Update Stats: No samples processed in update.", "yellow")

    def train(self):
        print_color("--- Starting PPO Training ---", "yellow")
        self.params.print_params()
        
        total_rollout_pbar = tqdm(range(self.params.num_rollouts), desc="Total Rollouts")
        for rollout_num in total_rollout_pbar:
            print_color(f"\n===== Rollout {rollout_num + 1} / {self.params.num_rollouts} =====", "blue")
            
            # 1. Collect experiences (rollouts)
            rollout_buffer = self.collect_rollout_experiences()
            if rollout_buffer is None:
                print_color("Failed to collect experiences. Stopping training.", "bold_red")
                break

            # 2. Update policy using collected experiences
            self.update_policy(rollout_buffer)

            # Logging (simple mean reward from this rollout)
            if (rollout_num + 1) % self.params.log_interval == 0:
                mean_reward = rollout_buffer["rewards"].mean().item()
                mean_value = rollout_buffer["values"].mean().item()
                mean_advantage = rollout_buffer["advantages"].mean().item() # Should be close to 0 if value func is good
                total_rollout_pbar.set_postfix({
                    "Mean Reward": f"{mean_reward:.2f}",
                    "Mean Value": f"{mean_value:.2f}",
                    "Mean Advantage": f"{mean_advantage:.2f}"
                })
                print_color(f"Rollout {rollout_num+1} Stats: Mean Reward={mean_reward:.3f}, Mean Value={mean_value:.3f}", "green")
            
            # Save model periodically
            if (rollout_num + 1) % self.params.save_interval == 0:
                self.save_model(rollout_num + 1)

        print_color("--- PPO Training Finished ---", "yellow")
        self.save_model("final")


    def save_model(self, checkpoint_name="final"):
        print_color(f"\nSaving model at checkpoint {checkpoint_name}...", "yellow")
        output_dir = os.path.join(self.params.save_path, f"checkpoint-{checkpoint_name}")
        os.makedirs(output_dir, exist_ok=True)

        # Save actor model (the part we are fine-tuning)
        self.model.actor_model.save_pretrained(os.path.join(output_dir, "actor"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "actor")) # Save tokenizer with actor

        # Save value head
        torch.save(self.model.value_head.state_dict(), os.path.join(output_dir, "value_head.pth"))
        
        # Save hyperparameters
        with open(os.path.join(output_dir, "ppo_hyperparameters.json"), 'w') as f:
            json.dump(self.params.__dict__, f, indent=4)
            
        print_color(f"Model saved to {output_dir}", "green")


def main():
    # --- Setup ---
    params = PPOHyperparameters()

    ## ENABLE THE BELOW COMMENTED OUT CODE IF YOU WANT TO TEST THE PIPELINE WORKS AND THE ITERATION SPEED. DEFAULT HYPERPARAMTERS 
    ## HAVE FOUND THE BE THE BEST FOR PROPER TRAINING RESULTS
    
    #  # --- Quick Test Hyperparameter Overrides ---
    # print_color("!!! APPLYING QUICK TEST HYPERPARAMETER OVERRIDES !!!", "bold_red")
    # params.max_input_length = 32         # Shorter input sequences
    # params.max_generated_length = 16     # Generate very short responses
    
    # params.num_rollouts = 2              # Only 2 full PPO iterations (rollout + update)
    # params.num_episodes_per_rollout = 4  # Collect only 4 prompt-response pairs per rollout
    # params.mini_batch_size = 2           # Use mini-batches of 2 for PPO updates (must be <= num_episodes_per_rollout)
    # params.ppo_epochs = 1                # Only 1 PPO optimization epoch on the collected data
    
    # # Optional: Force CPU if you want to avoid GPU involvement for a pure logic test,
    # # or if your GPU is slow for TinyLlama. Otherwise, it will use CUDA if available.
    # # params.device = "cpu"

    # # Optional: Reduce logging/saving intervals if you want to see those happen quickly
    # params.log_interval = 1
    # params.save_interval = 1 # Test model saving quickly
    # # --- End of Quick Test Overrides ---
    
    # Create dummy data file if it doesn't exist
    os.makedirs(os.path.dirname(params.data_file), exist_ok=True)
    if not os.path.exists(params.data_file):
        print_color(f"Data file {params.data_file} not found. Creating a dummy one.", "yellow")
        dummy_data = {
            "examples": [
                {"input": "What is John’s favorite color?", "response": "<memory_ask> What is John’s favorite color? </memory_ask>"},
                {"input": "John likes blue. Later, he mentions he also likes green.", "response": "<memory_write> John likes blue and green. </memory_write>"},
                {"input": "I don’t have the details about Mary’s phone number.", "response": "<memory_ask> What is Mary’s phone number? </memory_ask>"},
                {"input": "Record: Mary’s phone is 555-1234.", "response": "<memory_write> Mary’s phone is 555-1234. </memory_write>"},
                {"input": "What did I say about John's preferences?", "response": "<memory_ask> John's preferences </memory_ask>"},
                {"input": "Store this: The meeting is at 3 PM.", "response": "<memory_write> Meeting at 3 PM. </memory_write>"}
            ]
        }
        with open(params.data_file, 'w') as f:
            json.dump(dummy_data, f, indent=2)

    # Seed everything for reproducibility
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)

    # Load tokenizer
    tokenizer_name = params.tokenizer_name if params.tokenizer_name else params.model_name
    print_color(f"Loading tokenizer: {tokenizer_name}", "magenta")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='left')
    if tokenizer.pad_token is None:
        print_color("Tokenizer does not have a pad token. Setting pad_token = eos_token.", "yellow")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None: # Should not happen for most causal LMs
        raise ValueError("Tokenizer must have an EOS token for generation.")

    # Load dataset
    dataset = PromptDataset(params.data_file, tokenizer, params.max_input_length)
    if len(dataset) == 0:
        print_color("Dataset is empty. Exiting.", "bold_red")
        return

    # Initialize model
    # Pass tokenizer's eos_token_id to model for internal use if needed, e.g. value head logic
    model = ActorCritic(params.model_name, tokenizer.eos_token_id)

    # Initialize trainer
    trainer = PPOTrainer(model, tokenizer, params, dataset)

    # --- Train ---
    try:
        trainer.train()
    except Exception as e:
        print_color(f"An error occurred during training: {e}", "bold_red")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up CUDA memory if used
        if params.device == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()