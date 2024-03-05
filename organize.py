import json
import os
import numpy as np

# Path to your JSON file
json_path = '/mnt/disks/storage/data/finetune_data/4521k.json'

# Function to get image file size
def get_image_file_size(image_path):
    # Check if the file exists to avoid errors
    if os.path.exists(image_path):
        return os.path.getsize(image_path)  # Returns the file size in bytes
    else:
        return None

# Load JSON data
with open(json_path, 'r') as file:
    data = json.load(file)

# Initialize a list to store image file sizes
image_file_sizes = []

# Loop through each entry in the JSON data
for entry in data:
    if 'image' in entry:
        image_path = entry['image']
        file_size = get_image_file_size(image_path)
        if file_size is not None:
            image_file_sizes.append(file_size)
        else:
            print(f"Error: File {image_path} does not exist or cannot be accessed.")

# Define bins with more granularity, especially within the 15 MB range
# Bytes in a Megabyte (MB)
MB = 1024 * 1024

# Bins: < 1MB, 1-2MB, 2-3MB, ..., 14-15MB, > 15MB
bins = [i * MB for i in range(16)] + [max(image_file_sizes) + 1]  # Add an upper limit for > 15MB

# Calculate the distribution of image file sizes with the defined bins
file_size_distribution, bin_edges = np.histogram(image_file_sizes, bins=bins)

print("File size distribution:", file_size_distribution)
print("Bin edges:", bin_edges)
