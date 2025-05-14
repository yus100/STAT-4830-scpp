import json
import os
from typing import List, Dict, Any

class DataLoader:
    def __init__(self):
        """
        Initialize the data loader for RL training data
        """
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.datasets = {}
        self._load_datasets()

    def _load_datasets(self):
        """
        Load all JSON datasets from the data directory
        """
        for filename in ["d1.json", "d2.json"]:
            file_path = os.path.join(self.data_dir, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    self.datasets[filename] = data["examples"]
                    
            except FileNotFoundError:
                raise FileNotFoundError(f"Dataset file not found: {filename}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in file: {filename}")
            except KeyError:
                raise ValueError(f"Missing 'examples' key in dataset: {filename}")

    def get_dataset(self, name: str) -> List[Dict[str, str]]:
        """
        Get a specific dataset by name
        
        Args:
            name (str): Name of the dataset file (e.g., 'd1.json')
            
        Returns:
            List[Dict[str, str]]: List of examples with 'input' and 'response' keys
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset not found: {name}")
        return self.datasets[name]

    def get_all_examples(self) -> List[Dict[str, str]]:
        """
        Get all examples from all datasets combined
        
        Returns:
            List[Dict[str, str]]: Combined list of all examples
        """
        all_examples = []
        for dataset in self.datasets.values():
            all_examples.extend(dataset)
        return all_examples

    def get_example_count(self) -> Dict[str, int]:
        """
        Get the count of examples in each dataset
        
        Returns:
            Dict[str, int]: Dictionary mapping dataset names to example counts
        """
        return {name: len(examples) for name, examples in self.datasets.items()}