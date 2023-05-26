# Script to set up the environment. 
# Select the correct python interpreter.
# To create a new environment in python for this project
python -m venv env

# Activate the environment:

# For Windows
env/Scripts/activate

# For Mac
source env/bin/activate

# Install the requirements to get the libraries
pip install -r requirements.txt
pip3 install -r requirements.txt

# To run the app, try either commands
python main.py run
python3 main.py run