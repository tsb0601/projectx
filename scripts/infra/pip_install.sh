#!/bin/bash

# Navigate to the project directory
cd ~/projectx

# Install the current project in editable mode
pip install -e .

# Install additional dependencies for training
pip install -e '.[train]'

# Install specific versions of torch and torch_xla for TPU support
pip install torch==2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# Upgrade transformers to the latest version
pip install --upgrade transformers
 