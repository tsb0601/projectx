#!/bin/bash

# Directory where the disk will be mounted
MOUNT_POINT="/mnt/disks/storage"

# Device identifier for the disk to be mounted
DISK="/dev/sdb"

echo "Checking if disk is already mounted at '$MOUNT_POINT'..."

# Check if the disk is already mounted
if mount | grep -q "$MOUNT_POINT"; then
    echo "Disk is already mounted. Exiting."
    exit 0
else
    echo "Disk is not mounted. Proceeding with mounting process."
    # Create mount directory if it doesn't exist
    if [ ! -d "$MOUNT_POINT" ]; then
        echo "Creating mount directory at $MOUNT_POINT."
        sudo mkdir -p "$MOUNT_POINT"
    fi

    # Attempt to mount the disk
    echo "Mounting disk..."
    sudo mount -o ro,noload "$DISK" "$MOUNT_POINT" && echo "Mounted successfully at $MOUNT_POINT." || echo "Failed to mount disk."
fi
