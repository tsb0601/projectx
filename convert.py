import json

def convert_json_to_jsonl(input_json_path, output_jsonl_path):
    """
    Convert a JSON file to JSON Lines format.
    
    Parameters:
    - input_json_path: Path to the input JSON file.
    - output_jsonl_path: Path where the output JSONL file will be saved.
    """
    # Open the input JSON file
    with open(input_json_path, 'r') as json_file:
        data = json.load(json_file)  # Load the entire JSON array

    # Open the output JSONL file
    with open(output_jsonl_path, 'w') as jsonl_file:
        # Iterate over each object in the JSON array
        for item in data:
            # Convert each object to a JSON string + newline
            jsonl_file.write(json.dumps(item) + '\n')

# Example usage
input_json_path = '/mnt/disks/storage/data/finetune_data/4521k.json'  # Path to your large JSON file
output_jsonl_path = '4521kL.jsonl'  # Desired path for the JSONL file

convert_json_to_jsonl(input_json_path, output_jsonl_path)
