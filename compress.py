import json

# Define the input and output file names
input_file_name = '/mnt/disks/storage/data/finetune_data/4521k.json'
output_file_name = 'compress_4521k.json'

# Function to compress JSON data by removing unnecessary spaces
def compress_json_file(input_file_name, output_file_name):
    # Read the original JSON data
    with open(input_file_name, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    
    # Write the compressed JSON data
    with open(output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, separators=(',', ':'))

# Call the function to compress the JSON file
compress_json_file(input_file_name, output_file_name)
