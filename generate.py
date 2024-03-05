import json
import os

# Load the original JSON data
with open('/mnt/disks/storage/data/finetune_data/4521k.json', 'r') as file:
    data = json.load(file)

# # Bin edges from your histogram analysis
# bin_edges = [0, 1048576, 2097152, 5242880, 11534336, 12582912,
#              13631488, 14680064, 15728640]

bin_edges = [0, 5242880]

# Function to filter data by maximum file size, including entries without "image"
def filter_data_by_max_size(data, max_size):
    filtered_data = []
    for entry in data:
        # Include the entry if it does not have an "image" key
        if 'image' not in entry:
            filtered_data.append(entry)
        else:
            image_path = entry['image']
            # Check if the file exists and its size
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                if file_size <= max_size:
                    filtered_data.append(entry)
    return filtered_data

# Iterate through bin edges to create filtered datasets
for i, max_size in enumerate(bin_edges[1:], start=1):  # Skip the first bin edge (0)
    filtered_data = filter_data_by_max_size(data, max_size)
    file_name = f"{len(filtered_data)//1000}k.json"  # Naming convention
    with open(file_name, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)  # Use indent for pretty printing
    print(f"Created {file_name} with {len(filtered_data)} images up to {max_size} bytes.")
