# from huggingface_hub import HfApi
# api = HfApi()
# api.snapshot_download(repo_id='meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8', local_dir='~/Desktop/Development/models/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8', max_workers=2)

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Set model name and local save path
model_name = "microsoft/phi-2"
save_directory = "./phi-2-local"

# Load and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# Load and save the model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(save_directory)

print(f"Model saved locally in {save_directory}")