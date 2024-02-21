#!/bin/bash

# Array of lines to be added
LINES_TO_ADD=(
"export PATH=/mnt/disks/storage/envs/anaconda3/bin:\$PATH"
"export PJRT_DEVICE=TPU"
"export XLA_USE_BF16=1"
# "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/mnt/disks/storage/envs/anaconda3/lib
)

# File to which the lines will be added
FILE="$HOME/.bashrc"

# Function to append a line if it does not already exist in the file
append_line_if_not_exists() {
    local line="$1"
    local file="$2"
    grep -qF -- "$line" "$file" || echo "$line" >> "$file"
}

# Iterate over the array and append each line to ~/.bashrc if it does not already exist
for line in "${LINES_TO_ADD[@]}"; do
    append_line_if_not_exists "$line" "$FILE"
done

echo "Environment variables added to $FILE"

# Source the file to apply the changes to the current shell
source "$FILE"
echo "Environment variables sourced"
