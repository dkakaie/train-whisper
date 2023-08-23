#!/bin/bash

# Create and activate a python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip3 install -r requirements.txt

# Make a new directory
mkdir data

# Install git lfs and pull in large files
sudo apt-get install git-lfs
git lfs pull