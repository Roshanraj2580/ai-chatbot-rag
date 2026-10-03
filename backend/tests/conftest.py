"""
Pytest configuration file.
"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env variables for testing environment
from dotenv import load_dotenv
load_dotenv()

