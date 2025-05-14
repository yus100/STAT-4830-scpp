import json
import re

def separate_memory_tags(input_file, asks_file, writes_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    asks = []
    writes = []

    for example in data.get("examples", []):
        response = example.get("response", "")
        has_ask = "<memory_ask>" in response
        has_write = "<memory_write>" in response

        if has_ask:
            asks.append(example)
        if has_write:
            writes.append(example)

    with open(asks_file, 'w') as f:
        json.dump({"examples": asks}, f, indent=2)

    with open(writes_file, 'w') as f:
        json.dump({"examples": writes}, f, indent=2)

# Example usage
separate_memory_tags('data/syn_triplets_large/d1_4_combined.json', 'data/syn_dual/asks.json', 'data/syn_dual/writes.json')