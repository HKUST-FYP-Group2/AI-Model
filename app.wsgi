import sys
import os

# Set the correct project path
sys.path.insert(0, '/var/www/virtualwindow.cam/AI-Model')

# Activate virtual environment correctly
venv_path = "/var/www/virtualwindow.cam/AI-Model/.venv/bin/python"
if sys.executable != venv_path:
    os.execl(venv_path, venv_path, *sys.argv)

# Import the Flask app
from app import app as application
