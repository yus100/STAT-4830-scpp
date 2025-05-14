import os
import json

def aggregate_json_examples(input_directory, output_file):
    aggregated_examples = []

    # Traverse directory to find all .json files
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        examples = data.get("examples", [])
                        if not isinstance(examples, list):
                            print(f"Warning: 'examples' in {json_path} is not a list. Skipping.")
                            continue
                        aggregated_examples.extend(examples)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing {json_path}: {e}")

    # Write aggregated examples to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"examples": aggregated_examples}, f, indent=2, ensure_ascii=False)

    print(f"Aggregation complete. Total examples: {len(aggregated_examples)}")
    print(f"Output written to: {output_file}")

# Example usage
if __name__ == "__main__":
    input_dir = "writes/"
    output_path = "writes/aggregated.json"
    aggregate_json_examples(input_dir, output_path)