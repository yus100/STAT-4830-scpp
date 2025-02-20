import os
import json
import yaml
from typing import Any, Dict

class Config:
    """
    Singleton class to load and access configuration and prompts.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._config = None
        self._prompts = None
        self._load_config()
        self._load_prompts()

    def _load_config(self) -> None:
        """
        Load the configuration from config.json
        """
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r") as f:
            self._config = json.load(f)

    def _load_prompts(self) -> None:
        """
        Load the prompts from prompts.yaml
        """
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(prompts_path, "r") as f:
            self._prompts = yaml.safe_load(f)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary.

        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        if self._config is None:
            self._load_config()
        return self._config

    def get_prompt(self, key: str) -> str:
        """
        Get a prompt by its key.

        Args:
            key (str): The key of the prompt to retrieve.

        Returns:
            str: The prompt template string.

        Raises:
            KeyError: If the prompt key doesn't exist.
        """
        if self._prompts is None:
            self._load_prompts()
        
        if key not in self._prompts:
            raise KeyError(f"Prompt key '{key}' not found in prompts.yaml")
            
        return self._prompts[key]

# Create a singleton instance

if __name__ == "__main__":
    config = Config()
    print(config.get_prompt("rag_search"))