import json
import os

from Graph import Graph, Edge

class LGK(Graph):
    """
    Liquid Knowledge Graph
    """

    # read config file
    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "config")
        config_path = os.path.join(config_path, "config.json")
        with open(config_path, "r") as f:
            self._config = json.load(f)

    def __init__(self):
        super().__init__()

        self._load_config()


    async def preprocessing(self):
        pass

    async def postprocessing(self):
        
        passive_weak_ratio = self._config["weights"]["passive_weak_ratio"]
        for edge in self.edges:
            if not edge.permanent:
                edge.weight = edge.weight * passive_weak_ratio

    
    async def apply_edge_change_ratio(self, edge: Edge, ratio: float):
        if edge.permanent:
            return
        edge.weight = edge.weight * ratio