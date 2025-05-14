from huggingface_hub import hf_hub_download
import os

def download_model(model_id="meta-llama/Llama-2-3b-hf", filename="pytorch_model.bin"):
    """
    Downloads a model from Hugging Face Hub
    Args:
        model_id (str): The model ID on Hugging Face Hub
        filename (str): The specific file to download
    Returns:
        str: Path to the downloaded model file
    """
    try:
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            token=os.getenv("HF_TOKEN")  # Make sure to set your HF_TOKEN environment variable
        )
        print(f"Model downloaded successfully to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

# Example usage
if __name__ == "__main__":
    model_path = download_model()



# other model for reference

# from huggingface_hub import HfApi
# api = HfApi()
# api.snapshot_download(repo_id='meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8', local_dir='~/Desktop/Development/models/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8', max_workers=2)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import os

# # Set model name and local save path
# model_name = "microsoft/phi-2"
# save_directory = "./phi-2-local"

# # Load and save the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(save_directory)

# # Load and save the model
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.save_pretrained(save_directory)

# print(f"Model saved locally in {save_directory}")