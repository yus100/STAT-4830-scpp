import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from data_loader import DataLoader
from typing import List, Dict, Any, Tuple
import numpy as np
from datasets import Dataset
import wandb
import os
from tqdm import tqdm
import time

class MemoryAgent:
    def __init__(
        self,
        model_name: str = "llama3.2",
        model_type: str = "llama",
        learning_rate: float = 1e-5,
        max_steps: int = 1000,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4
    ):
        self.model_name = model_name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_safetensors=True,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            use_safetensors=True,
            model_type=model_type,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Create reference model for KL penalty
        self.ref_model = create_reference_model(self.model)
        
        # PPO config
        self.ppo_config = PPOConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=0.1,
            max_steps=max_steps,
            log_with="wandb"
        )
        
        # Initialize trainer
        self.trainer = None
        
    def _prepare_dataset(self, examples: List[Dict[str, str]]) -> Dataset:
        """Convert examples to HuggingFace Dataset format"""
        formatted_data = []
        
        for example in examples:
            formatted_data.append({
                "input": example["input"],
                "response": example["response"],
                "query": f"Given the input: {example['input']}\nGenerate an appropriate response:"
            })
            
        return Dataset.from_list(formatted_data)
    
    def _compute_reward(
        self,
        response: str,
        target: str
    ) -> float:
        """
        Compute reward for generated response
        
        Rewards:
        - +1 for correct use of memory tokens when needed
        - -1 for incorrect use of memory tokens
        - -0.5 for missing memory tokens when needed
        - +0.5 for correctly not using memory tokens
        """
        
        reward = 0.0
        
        # Check if target contains memory tokens
        target_has_memory = "<memory_" in target
        response_has_memory = "<memory_" in response
        
        if target_has_memory and response_has_memory:
            # Check if the right type of memory token is used
            target_token = "write" if "<memory_write>" in target else "read"
            response_token = "write" if "<memory_write>" in response else "read"
            
            if target_token == response_token:
                reward += 1.0
            else:
                reward -= 1.0
                
        elif target_has_memory and not response_has_memory:
            reward -= 0.5
        elif not target_has_memory and not response_has_memory:
            reward += 0.5
        else:  # not target_has_memory and response_has_memory
            reward -= 1.0
            
        # Add small KL penalty to prevent drift
        kl_penalty = -0.1
        reward += kl_penalty
            
        return reward
    
    def _tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Tokenize inputs for the model"""
        return self.tokenizer(
            examples["query"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    def train(self, train_dataset: Dataset) -> None:
        """Train the model using PPO"""
        
        print("\n=== Starting Training ===")
        print(f"Model: {self.model_name}")
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Batch size: {self.ppo_config.batch_size}")
        print(f"Max steps: {self.ppo_config.max_steps}")
        print("=======================\n")
        
        # Initialize wandb
        wandb.init(project="memory-agent-training", name="ppo_training")
        
        # Tokenize dataset
        print("Tokenizing dataset...")
        train_dataset = train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        print("Tokenization complete!")
        
        # Initialize trainer
        print("Initializing PPO trainer...")
        self.trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=train_dataset
        )
        print("Trainer initialized!")
        
        # Training loop
        progress_bar = tqdm(range(self.ppo_config.max_steps), desc="Training Progress")
        total_rewards = []
        start_time = time.time()
        
        for epoch in progress_bar:
            batch = next(iter(self.trainer.dataloader))
            query_tensors = batch["input_ids"]
            
            # Generate responses
            progress_bar.set_description(f"Epoch {epoch}: Generating responses")
            response_tensors = self.trainer.generate(
                query_tensors,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.9
            )
            
            # Decode responses
            progress_bar.set_description(f"Epoch {epoch}: Decoding responses")
            responses = self.tokenizer.batch_decode(response_tensors)
            batch_responses = [resp.split("Generate an appropriate response:")[-1].strip() for resp in responses]
            
            # Compute rewards
            progress_bar.set_description(f"Epoch {epoch}: Computing rewards")
            rewards = [
                self._compute_reward(response, example["response"])
                for response, example in zip(batch_responses, batch)
            ]
            total_rewards.extend(rewards)
            
            # Run PPO step
            progress_bar.set_description(f"Epoch {epoch}: Running PPO step")
            stats = self.trainer.step(query_tensors, response_tensors, rewards)
            
            # Update progress bar
            avg_reward = np.mean(rewards)
            progress_bar.set_postfix({
                'reward': f"{avg_reward:.3f}",
                'policy_loss': f"{stats['policy_loss']:.3f}",
                'value_loss': f"{stats['value_loss']:.3f}",
                'kl': f"{stats['avg_kl']:.3f}"
            })
            
            # Log metrics
            wandb.log(
                {
                    "reward": avg_reward,
                    "policy_loss": stats["policy_loss"],
                    "value_loss": stats["value_loss"],
                    "avg_kl": stats["avg_kl"],
                    "epoch": epoch
                }
            )
            
            # Early stopping check
            if stats["avg_kl"] > self.ppo_config.target_kl:
                print(f"\n⚠️ Early stopping at epoch {epoch} due to high KL divergence")
                break
        
        # Training summary
        training_time = time.time() - start_time
        print("\n=== Training Summary ===")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Average reward: {np.mean(total_rewards):.3f}")
        print(f"Final KL divergence: {stats['avg_kl']:.3f}")
        print("=====================\n")

    def save_model(self, path: str) -> None:
        """Save the fine-tuned model"""
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def generate(self, input_text: str) -> str:
        """Generate response for given input"""
        query = f"Given the input: {input_text}\nGenerate an appropriate response:"
        inputs = self.tokenizer(query, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=128,
                do_sample=True,
                temperature=0.9
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Generate an appropriate response:")[-1].strip()

def main():
    print("\n🚀 Starting Memory Agent Training Pipeline")
    
    # Initialize wandb
    print("Logging into wandb...")
    wandb.login()
    
    # Load datasets
    print("\nLoading datasets...")
    loader = DataLoader()
    d1 = loader.get_dataset("d1.json")
    d2 = loader.get_dataset("d2.json")
    print(f"Loaded d1.json ({len(d1)} examples)")
    print(f"Loaded d2.json ({len(d2)} examples)")
    
    # Combine datasets
    all_examples = d1 + d2
    print(f"Total examples: {len(all_examples)}")
    
    # Initialize agent
    print("\nInitializing Memory Agent...")
    agent = MemoryAgent(
        model_name="models/llama-3.2-3B-instruct",
        learning_rate=1e-5,
        max_steps=1000,
        batch_size=4,
        gradient_accumulation_steps=4
    )
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = agent._prepare_dataset(all_examples)
    print("Dataset preparation complete!")
    
    # Train
    print("\n📈 Starting training process...")
    agent.train(train_dataset)
    
    # Save model
    print("\n💾 Saving trained model...")
    agent.save_model("memory_agent_model")
    print("Model saved successfully!")
    
    # Test generation
    print("\n🧪 Running test generations...")
    test_inputs = [
        "What is John's favorite color?",
        "John likes blue. Later, he mentions he also likes green.",
        "I am just sharing my thoughts about the book."
    ]
    
    print("\nTest Results:")
    print("=============")
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\nTest {i}:")
        print(f"Input: {input_text}")
        response = agent.generate(input_text)
        print(f"Response: {response}")
        print("-" * 50)
    
    # Close wandb
    print("\n👋 Closing wandb session...")
    wandb.finish()
    print("\n✨ Training pipeline completed!")

if __name__ == "__main__":
    main()