import json

# Load the JSON data from the 4706k.json file
input_file_name = '/mnt/disks/storage/data/finetune_data/4521k.json'
with open(input_file_name, 'r') as f:
    data = json.load(f)

# Filter out entries where the image path starts with "llavar/finetune/0xxxxx"
filtered_data = [entry for entry in data if not entry.get('image', '').startswith('vizwiz')]

# Count the remaining number of entries
remaining_entries_count = len(filtered_data)

# Create a new file name based on the number of remaining entries
output_file_name = f"{remaining_entries_count//1000}k.json"

# Save the filtered data to the new file
with open(output_file_name, 'w') as outfile:
    json.dump(filtered_data, outfile, indent=4)

print(f"Filtered data saved as {output_file_name} with {remaining_entries_count} entries.")
