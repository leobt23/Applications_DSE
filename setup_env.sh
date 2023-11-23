#!/bin/bash
### source dsapps_env/bin/activate
# Define the environment name
ENV_NAME="dsapps_env"

# Create a virtual environment
echo "Creating virtual environment named $ENV_NAME"
python3 -m venv $ENV_NAME

# Activate the virtual environment
echo "Activating virtual environment"
source $ENV_NAME/bin/activate

# Install requirements
echo "Installing requirements from requirements.txt"
pip install -r requirements.txt

echo "Setup complete. Virtual environment '$ENV_NAME' is ready."
